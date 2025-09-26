// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Conversions between PyArrow and DataFusion types

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use arrow::array::{Array, ArrayData};
use arrow::error::ArrowError;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use pyo3::exceptions::{
    PyException, PyNotImplementedError, PyRuntimeError, PyValueError,
};
use pyo3::prelude::PyErr;
use pyo3::types::{PyAnyMethods, PyList};
use pyo3::{Bound, FromPyObject, IntoPyObject, PyAny, PyObject, PyResult, Python};

use crate::error::GenericError;
use crate::{DataFusionError, ScalarValue};

impl From<DataFusionError> for PyErr {
    fn from(err: DataFusionError) -> PyErr {
        match try_extract_python_error(err) {
            PythonErrorExtraction::Python(py_err) => py_err,
            PythonErrorExtraction::Other(err) => map_datafusion_error(err),
        }
    }
}

/// Wrapper around [`PyErr`] which implements [`Error`].
#[derive(Debug)]
pub struct PythonError {
    inner: PyErr,
}

impl PythonError {
    /// Create a new [`PythonError`] wrapper.
    pub fn new(err: PyErr) -> Self {
        Self { inner: err }
    }

    /// Consume this wrapper and return the contained [`PyErr`].
    pub fn into_inner(self) -> PyErr {
        self.inner
    }

    /// Borrow the contained [`PyErr`].
    pub fn inner_ref(&self) -> &PyErr {
        &self.inner
    }
}

impl Display for PythonError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl Error for PythonError {}

impl From<PyErr> for DataFusionError {
    fn from(err: PyErr) -> Self {
        DataFusionError::External(Box::new(PythonError::new(err)))
    }
}

enum PythonErrorExtraction<T> {
    Python(PyErr),
    Other(T),
}

fn try_extract_python_error(
    err: DataFusionError,
) -> PythonErrorExtraction<DataFusionError> {
    // helper to convert PythonErrorExtraction<Inner> -> PythonErrorExtraction<DataFusionError>
    fn map_other<T>(
        res: PythonErrorExtraction<T>,
        wrap: impl FnOnce(T) -> DataFusionError,
    ) -> PythonErrorExtraction<DataFusionError> {
        match res {
            PythonErrorExtraction::Python(py) => PythonErrorExtraction::Python(py),
            PythonErrorExtraction::Other(inner) => {
                PythonErrorExtraction::Other(wrap(inner))
            }
        }
    }

    match err {
        DataFusionError::External(generic) => {
            map_other(try_extract_python_error_from_generic_error(generic), |e| {
                DataFusionError::External(e)
            })
        }
        DataFusionError::ArrowError(boxed, backtrace) => {
            map_other(try_extract_python_error_from_arrow_error(*boxed), |e| {
                DataFusionError::ArrowError(Box::new(e), backtrace)
            })
        }
        DataFusionError::Context(ctx, inner) => {
            map_other(try_extract_python_error(*inner), |e| {
                DataFusionError::Context(ctx, Box::new(e))
            })
        }
        DataFusionError::Diagnostic(diag, inner) => {
            map_other(try_extract_python_error(*inner), |e| {
                DataFusionError::Diagnostic(diag, Box::new(e))
            })
        }
        DataFusionError::Collection(errors) => {
            let mut rebuilt = Vec::with_capacity(errors.len());
            for error in errors {
                match try_extract_python_error(error) {
                    PythonErrorExtraction::Python(py_err) => {
                        return PythonErrorExtraction::Python(py_err)
                    }
                    PythonErrorExtraction::Other(other) => rebuilt.push(other),
                }
            }
            PythonErrorExtraction::Other(DataFusionError::Collection(rebuilt))
        }
        DataFusionError::Shared(shared) => match Arc::try_unwrap(shared) {
            Ok(inner) => try_extract_python_error(inner),
            Err(shared) => PythonErrorExtraction::Other(DataFusionError::Shared(shared)),
        },
        other => PythonErrorExtraction::Other(other),
    }
}

fn try_extract_python_error_from_arrow_error(
    err: ArrowError,
) -> PythonErrorExtraction<ArrowError> {
    match err {
        ArrowError::ExternalError(error) => {
            match try_extract_python_error_from_generic_error(error) {
                PythonErrorExtraction::Python(py_err) => {
                    PythonErrorExtraction::Python(py_err)
                }
                PythonErrorExtraction::Other(error) => {
                    PythonErrorExtraction::Other(ArrowError::ExternalError(error))
                }
            }
        }
        other => PythonErrorExtraction::Other(other),
    }
}

fn try_extract_python_error_from_generic_error(
    error: GenericError,
) -> PythonErrorExtraction<GenericError> {
    let error = match error.downcast::<PyErr>() {
        Ok(py_err) => {
            return PythonErrorExtraction::Python(*py_err);
        }
        Err(error) => error,
    };

    let error = match error.downcast::<Arc<PyErr>>() {
        Ok(py_err) => match Arc::try_unwrap(*py_err) {
            Ok(py_err) => {
                return PythonErrorExtraction::Python(py_err);
            }
            Err(py_err) => {
                return PythonErrorExtraction::Other(Box::new(py_err));
            }
        },
        Err(error) => error,
    };

    let error = match error.downcast::<PythonError>() {
        Ok(python_error) => {
            return PythonErrorExtraction::Python(python_error.into_inner())
        }
        Err(error) => error,
    };

    let error = match error.downcast::<DataFusionError>() {
        Ok(datafusion_error) => match try_extract_python_error(*datafusion_error) {
            PythonErrorExtraction::Python(py_err) => {
                return PythonErrorExtraction::Python(py_err)
            }
            PythonErrorExtraction::Other(other) => {
                return PythonErrorExtraction::Other(Box::new(other))
            }
        },
        Err(error) => error,
    };

    let error = match error.downcast::<ArrowError>() {
        Ok(arrow_error) => {
            match try_extract_python_error_from_arrow_error(*arrow_error) {
                PythonErrorExtraction::Python(py_err) => {
                    return PythonErrorExtraction::Python(py_err)
                }
                PythonErrorExtraction::Other(other) => {
                    return PythonErrorExtraction::Other(Box::new(other))
                }
            }
        }
        Err(error) => error,
    };

    let error = match error.downcast::<Arc<DataFusionError>>() {
        Ok(datafusion_error) => match Arc::try_unwrap(*datafusion_error) {
            Ok(inner) => match try_extract_python_error(inner) {
                PythonErrorExtraction::Python(py_err) => {
                    return PythonErrorExtraction::Python(py_err)
                }
                PythonErrorExtraction::Other(other) => {
                    return PythonErrorExtraction::Other(Box::new(other))
                }
            },
            Err(datafusion_error) => {
                return PythonErrorExtraction::Other(Box::new(datafusion_error))
            }
        },
        Err(error) => error,
    };

    let error = match error.downcast::<Arc<ArrowError>>() {
        Ok(arrow_error) => match Arc::try_unwrap(*arrow_error) {
            Ok(inner) => match try_extract_python_error_from_arrow_error(inner) {
                PythonErrorExtraction::Python(py_err) => {
                    return PythonErrorExtraction::Python(py_err)
                }
                PythonErrorExtraction::Other(other) => {
                    return PythonErrorExtraction::Other(Box::new(other))
                }
            },
            Err(arrow_error) => {
                return PythonErrorExtraction::Other(Box::new(arrow_error))
            }
        },
        Err(error) => error,
    };

    let error = match error.downcast::<Arc<PythonError>>() {
        Ok(python_error) => match Arc::try_unwrap(*python_error) {
            Ok(python_error) => {
                return PythonErrorExtraction::Python(python_error.into_inner())
            }
            Err(python_error) => {
                return PythonErrorExtraction::Other(Box::new(python_error))
            }
        },
        Err(error) => error,
    };

    PythonErrorExtraction::Other(error)
}

fn map_datafusion_error(err: DataFusionError) -> PyErr {
    match err {
        DataFusionError::Plan(message) | DataFusionError::Configuration(message) => {
            PyValueError::new_err(message)
        }
        DataFusionError::Execution(message) => PyRuntimeError::new_err(message),
        DataFusionError::NotImplemented(message) => {
            PyNotImplementedError::new_err(message)
        }
        other => PyException::new_err(other.to_string()),
    }
}

impl FromPyArrow for ScalarValue {
    fn from_pyarrow_bound(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = value.py();
        let typ = value.getattr("type")?;
        let val = value.call_method0("as_py")?;

        // construct pyarrow array from the python value and pyarrow type
        let factory = py.import("pyarrow")?.getattr("array")?;
        let args = PyList::new(py, [val])?;
        let array = factory.call1((args, typ))?;

        // convert the pyarrow array to rust array using C data interface
        let array = arrow::array::make_array(ArrayData::from_pyarrow_bound(&array)?);
        let scalar = ScalarValue::try_from_array(&array, 0)?;

        Ok(scalar)
    }
}

impl ToPyArrow for ScalarValue {
    fn to_pyarrow(&self, py: Python) -> PyResult<PyObject> {
        let array = self.to_array()?;
        // convert to pyarrow array using C data interface
        let pyarray = array.to_data().to_pyarrow(py)?;
        let pyscalar = pyarray.call_method1(py, "__getitem__", (0,))?;

        Ok(pyscalar)
    }
}

impl<'source> FromPyObject<'source> for ScalarValue {
    fn extract_bound(value: &Bound<'source, PyAny>) -> PyResult<Self> {
        Self::from_pyarrow_bound(value)
    }
}

impl<'source> IntoPyObject<'source> for ScalarValue {
    type Target = PyAny;

    type Output = Bound<'source, Self::Target>;

    type Error = PyErr;

    fn into_pyobject(self, py: Python<'source>) -> Result<Self::Output, Self::Error> {
        let array = self.to_array()?;
        // convert to pyarrow array using C data interface
        let pyarray = array.to_data().to_pyarrow(py)?;
        let pyarray_bound = pyarray.bind(py);
        pyarray_bound.call_method1("__getitem__", (0,))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pyo3::ffi::c_str;
    use pyo3::prepare_freethreaded_python;
    use pyo3::py_run;
    use pyo3::types::{PyDict, PyDictMethods, PyStringMethods, PyTypeMethods};
    use pyo3::PyTypeInfo;

    use super::*;

    fn init_python() {
        prepare_freethreaded_python();
        Python::with_gil(|py| {
            if py.run(c_str!("import pyarrow"), None, None).is_err() {
                let locals = PyDict::new(py);
                py.run(
                    c_str!(
                        "import sys; executable = sys.executable; python_path = sys.path"
                    ),
                    None,
                    Some(&locals),
                )
                .expect("Couldn't get python info");
                let executable = locals
                    .get_item("executable")
                    .expect("failed to get python executable entry")
                    .expect("python executable missing");
                let executable: String = executable
                    .extract()
                    .expect("python executable should be a string");

                let python_path = locals
                    .get_item("python_path")
                    .expect("failed to get python path entry")
                    .expect("python path missing");
                let python_path: Vec<String> = python_path
                    .extract()
                    .expect("python path should be a list of strings");

                panic!("pyarrow not found\nExecutable: {executable}\nPython path: {python_path:?}\n\
                         HINT: try `pip install pyarrow`\n\
                         NOTE: On Mac OS, you must compile against a Framework Python \
                         (default in python.org installers and brew, but not pyenv)\n\
                         NOTE: On Mac OS, PYO3 might point to incorrect Python library \
                         path when using virtual environments. Try \
                         `export PYTHONPATH=$(python -c \"import sys; print(sys.path[-1])\")`\n")
            }
        })
    }

    #[test]
    fn python_error_round_trip_plain_pyerr() {
        prepare_freethreaded_python();

        Python::with_gil(|py| {
            let df_err =
                DataFusionError::External(Box::new(PyValueError::new_err("plain pyerr")));
            let py_err = PyErr::from(df_err);

            assert!(py_err.get_type(py).is(PyValueError::type_object(py)));

            let value = py_err.value(py);
            let binding = value
                .str()
                .expect("python error has a string representation");
            let message = binding
                .to_str()
                .expect("python error message should be valid UTF-8");
            assert_eq!(message, "plain pyerr");
        });
    }

    #[test]
    fn python_error_round_trip_arc_pyerr() {
        prepare_freethreaded_python();

        Python::with_gil(|py| {
            let df_err = DataFusionError::External(Box::new(Arc::new(
                PyValueError::new_err("arc pyerr"),
            )));
            let py_err = PyErr::from(df_err);

            assert!(py_err.get_type(py).is(PyValueError::type_object(py)));

            let value = py_err.value(py);
            let binding = value
                .str()
                .expect("python error has a string representation");
            let message = binding
                .to_str()
                .expect("python error message should be valid UTF-8");
            assert_eq!(message, "arc pyerr");
        });
    }

    #[test]
    fn test_roundtrip() {
        init_python();

        let example_scalars = vec![
            ScalarValue::Boolean(Some(true)),
            ScalarValue::Int32(Some(23)),
            ScalarValue::Float64(Some(12.34)),
            ScalarValue::from("Hello!"),
            ScalarValue::Date32(Some(1234)),
        ];

        Python::with_gil(|py| {
            for scalar in example_scalars.iter() {
                let result = ScalarValue::from_pyarrow_bound(
                    scalar.to_pyarrow(py).unwrap().bind(py),
                )
                .unwrap();
                assert_eq!(scalar, &result);
            }
        });
    }

    #[test]
    fn test_py_scalar() -> PyResult<()> {
        init_python();

        Python::with_gil(|py| -> PyResult<()> {
            let scalar_float = ScalarValue::Float64(Some(12.34));
            let py_float = scalar_float
                .into_pyobject(py)?
                .call_method0("as_py")
                .unwrap();
            py_run!(py, py_float, "assert py_float == 12.34");

            let scalar_string = ScalarValue::Utf8(Some("Hello!".to_string()));
            let py_string = scalar_string
                .into_pyobject(py)?
                .call_method0("as_py")
                .unwrap();
            py_run!(py, py_string, "assert py_string == 'Hello!'");

            Ok(())
        })
    }

    #[test]
    fn python_error_roundtrip_preserves_traceback() {
        prepare_freethreaded_python();

        Python::with_gil(|py| -> PyResult<()> {
            let json = py.import("json")?;
            let err = json.call_method1("loads", ("{",)).unwrap_err();
            let error_type = err.get_type(py).name()?;
            let error_type = error_type.to_str()?.to_string();
            let message = err.value(py).str()?.to_str()?.to_string();
            let original_trace = err.traceback(py).map(|tb| tb.as_ptr());

            let py_err = PyErr::from(DataFusionError::from(err));

            let roundtrip_type = py_err.get_type(py).name()?;
            let roundtrip_type = roundtrip_type.to_str()?.to_string();
            assert_eq!(error_type, roundtrip_type);

            let roundtrip_message = py_err.value(py).str()?.to_str()?.to_string();
            assert_eq!(message, roundtrip_message);

            let roundtrip_trace = py_err.traceback(py).map(|tb| tb.as_ptr());
            assert_eq!(original_trace, roundtrip_trace);

            Ok(())
        })
        .expect("python roundtrip test failed");
    }

    #[test]
    fn python_error_roundtrip_through_context() {
        prepare_freethreaded_python();

        Python::with_gil(|py| -> PyResult<()> {
            let importlib = py.import("importlib")?;
            let err = importlib
                .call_method1("import_module", ("does_not_exist_module",))
                .unwrap_err();
            let err_type = err.get_type(py).name()?;
            let err_type = err_type.to_str()?.to_string();
            let message = err.value(py).str()?.to_str()?.to_string();
            let original_trace = err.traceback(py).map(|tb| tb.as_ptr());

            let df_err = DataFusionError::Context(
                "while executing python callback".to_string(),
                Box::new(DataFusionError::from(err)),
            );
            let py_err = PyErr::from(df_err);

            let roundtrip_type = py_err.get_type(py).name()?;
            let roundtrip_type = roundtrip_type.to_str()?.to_string();
            assert_eq!(err_type, roundtrip_type);

            let roundtrip_message = py_err.value(py).str()?.to_str()?.to_string();
            assert_eq!(message, roundtrip_message);

            let roundtrip_trace = py_err.traceback(py).map(|tb| tb.as_ptr());
            assert_eq!(original_trace, roundtrip_trace);

            Ok(())
        })
        .expect("python error context roundtrip test failed");
    }

    #[test]
    fn python_error_roundtrip_through_arrow() {
        prepare_freethreaded_python();

        Python::with_gil(|py| -> PyResult<()> {
            let base64 = py.import("base64")?;
            let err = base64.call_method1("b64decode", ("$",)).unwrap_err();
            let err_type = err.get_type(py).name()?.to_str()?.to_string();
            let message = err.value(py).str()?.to_str()?.to_string();
            let original_trace = err.traceback(py).map(|tb| tb.as_ptr());

            let df_err = DataFusionError::ArrowError(
                Box::new(ArrowError::ExternalError(Box::new(DataFusionError::from(
                    err,
                )))),
                None,
            );
            let py_err = PyErr::from(df_err);

            let roundtrip_type = py_err.get_type(py).name()?.to_str()?.to_string();
            assert_eq!(err_type, roundtrip_type);

            let roundtrip_message = py_err.value(py).str()?.to_str()?.to_string();
            assert_eq!(message, roundtrip_message);

            let roundtrip_trace = py_err.traceback(py).map(|tb| tb.as_ptr());
            assert_eq!(original_trace, roundtrip_trace);

            Ok(())
        })
        .expect("python error arrow roundtrip test failed");
    }

    #[test]
    fn python_error_roundtrip_through_nested_wrappers() {
        prepare_freethreaded_python();

        Python::with_gil(|py| -> PyResult<()> {
            let asyncio = py.import("asyncio")?;
            let err = asyncio.call_method0("get_running_loop").unwrap_err();
            let err_type = err.get_type(py).name()?.to_str()?.to_string();
            let message = err.value(py).str()?.to_str()?.to_string();

            let df_err = DataFusionError::ArrowError(
                Box::new(ArrowError::ExternalError(Box::new(
                    ArrowError::ExternalError(Box::new(DataFusionError::from(err))),
                ))),
                None,
            );

            let df_err = DataFusionError::Context(
                "while executing python UDF".to_string(),
                Box::new(DataFusionError::Collection(vec![
                    DataFusionError::Internal("ignore me".to_string()),
                    df_err,
                ])),
            );

            let py_err = PyErr::from(df_err);

            let roundtrip_type = py_err.get_type(py).name()?.to_str()?.to_string();
            assert_eq!(err_type, roundtrip_type);

            let roundtrip_message = py_err.value(py).str()?.to_str()?.to_string();
            assert_eq!(message, roundtrip_message);

            Ok(())
        })
        .expect("python error nested wrapper roundtrip test failed");
    }
}
