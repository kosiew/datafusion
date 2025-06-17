use super::*;
use bigdecimal::{BigDecimal, ToPrimitive};
use datafusion_catalog::MemTable;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::str::FromStr;

#[tokio::test]
async fn decimal_multiplication_sum() -> Result<()> {
    let precision = 38u8;
    let scale = 10i8;
    let factor = BigDecimal::from(10u64.pow(scale as u32));
    let mut rng = StdRng::seed_from_u64(42);
    const ROWS: usize = 100_000;
    let mut d1_vals = Vec::with_capacity(ROWS);
    let mut d2_vals = Vec::with_capacity(ROWS);
    let mut expected = BigDecimal::from(0);
    for _ in 0..ROWS {
        let n1: u64 = rng.random_range(1..=(1u64 << 53));
        let p = 10u64.pow(rng.random_range(1..=8));
        let d1 = BigDecimal::from(n1) / BigDecimal::from(p);
        let n2: u64 = rng.random_range(1..=100);
        let d2 = BigDecimal::from_str(&format!("0.{}", n2)).unwrap();
        expected += &d1 * &d2;
        let d1_unscaled = ((&d1 * &factor).with_scale(0)).to_i128().unwrap();
        let d2_unscaled = ((&d2 * &factor).with_scale(0)).to_i128().unwrap();
        d1_vals.push(d1_unscaled);
        d2_vals.push(d2_unscaled);
    }
    let arr1 = Decimal128Array::from_iter_values(d1_vals)
        .with_precision_and_scale(precision, scale)
        .unwrap();
    let arr2 = Decimal128Array::from_iter_values(d2_vals)
        .with_precision_and_scale(precision, scale)
        .unwrap();
    let schema = Schema::new(vec![
        Field::new("d1", DataType::Decimal128(precision, scale), false),
        Field::new("d2", DataType::Decimal128(precision, scale), false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(arr1) as ArrayRef, Arc::new(arr2) as ArrayRef],
    )?;
    let memtable = MemTable::try_new(Arc::new(schema), vec![vec![batch]])?;
    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(memtable))?;
    let df = ctx.sql("SELECT sum(d1 * d2) FROM t").await?;
    let results = df.collect().await?;
    assert_eq!(results.len(), 1);
    let value = results[0]
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .unwrap()
        .value(0);
    let expected_unscaled = ((expected * factor).with_scale(0)).to_i128().unwrap();
    assert_eq!(value, expected_unscaled);
    Ok(())
}
