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
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use pyo3::exceptions::{
    PyException, PyNotImplementedError, PyRuntimeError, PyValueError,
};
use pyo3::prelude::PyErr;
use pyo3::types::{PyAnyMethods, PyList};
use pyo3::{Bound, FromPyObject, IntoPyObject, PyAny, PyObject, PyResult, Python};

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
    pub fn as_ref(&self) -> &PyErr {
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

enum PythonErrorExtraction {
    Python(PyErr),
    Other(DataFusionError),
}

fn try_extract_python_error(err: DataFusionError) -> PythonErrorExtraction {
    match err {
        DataFusionError::External(error) => match error.downcast::<PythonError>() {
            Ok(python_error) => PythonErrorExtraction::Python(python_error.into_inner()),
            Err(error) => PythonErrorExtraction::Other(DataFusionError::External(error)),
        },
        DataFusionError::Context(context, inner) => {
            match try_extract_python_error(*inner) {
                PythonErrorExtraction::Python(py_err) => {
                    PythonErrorExtraction::Python(py_err)
                }
                PythonErrorExtraction::Other(inner) => PythonErrorExtraction::Other(
                    DataFusionError::Context(context, Box::new(inner)),
                ),
            }
        }
        DataFusionError::Diagnostic(diagnostic, inner) => {
            match try_extract_python_error(*inner) {
                PythonErrorExtraction::Python(py_err) => {
                    PythonErrorExtraction::Python(py_err)
                }
                PythonErrorExtraction::Other(inner) => PythonErrorExtraction::Other(
                    DataFusionError::Diagnostic(diagnostic, Box::new(inner)),
                ),
            }
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
    use pyo3::ffi::c_str;
    use pyo3::prepare_freethreaded_python;
    use pyo3::py_run;
    use pyo3::types::{PyDict, PyDictMethods, PyStringMethods, PyTypeMethods};

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
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
def intermediary():
    raise ValueError("boom from python")

def wrapper():
    intermediary()
"#
                ),
                None,
                Some(&locals),
            )?;

            let wrapper = locals
                .get_item("wrapper")?
                .expect("wrapper function missing");
            let err = wrapper.call0().unwrap_err();
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
            let locals = PyDict::new(py);
            py.run(
                pyo3::ffi::c_str!(
                    r#"
def raises_runtime_error():
    raise RuntimeError("context preserved")
"#
                ),
                None,
                Some(&locals),
            )?;

            let raises_runtime_error = locals
                .get_item("raises_runtime_error")?
                .expect("raises_runtime_error missing");
            let err = raises_runtime_error.call0().unwrap_err();
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
}
