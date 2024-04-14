use ndarray::prelude::*;
use crate::types::*;

fn exp<D: Dimension>(x: Array<NNFloat, D>) -> Array<NNFloat, D> {
    x.mapv_into(|value | { value.exp() })
}

pub fn sigmoid<D: Dimension>(x: &Array<NNFloat, D>) -> Array<NNFloat, D> {
    1.0 / (1.0 + exp(-x))
}


pub fn softmax(x: &NNMatrix) -> NNMatrix {
    // P.S: use `broadcast()` + reshape to avoid using `t()` twice ?
    let x = x.t().to_owned();
    let x_max = x.fold_axis(Axis(0),
                            0.0,
                           |acc, v| { if *acc < *v { *v } else { *acc } });
    let x = exp(x - x_max);
    let x_sum = x.sum_axis(Axis(0));
    let y = x / x_sum;

    y.t().to_owned()
}

pub fn cross_entropy_error(y: &NNMatrix, t: &NNMatrix) -> NNFloat {
    let batch_size = y.shape()[0];
    let k = if y.len() == t.len() {
        // one hot vector
        t * y.mapv(|v| { (v + 1e-7).ln() })
    } else {
        let mut d = Array::<NNFloat, _>::zeros(t.raw_dim());
        for (i, arr) in t.axis_iter(Axis(0)).enumerate() {
            for (j, pos) in arr.iter().enumerate() {
                d[[i, j]] = (y[[i, *pos as usize]] + 1e-7).ln();
            }
        }
        d
    };
    -k.sum() / (batch_size as NNFloat)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array};

    #[test]
    fn test_sigmoid() {
        let x = array![-1.0, 1.0, 2.0];
        let x_results = array![0.26894142, 0.73105858, 0.880797];
        assert_eq!(sigmoid(&x), x_results);

        let x = array![
            [-1.0, 1.0, 2.0]
            ,[-5.0, 5.0, 0.1]];
        let x_results = array![
            [0.26894142, 0.73105858, 0.880797]
            , [0.006692851, 0.99330715, 0.52497919]];
        assert_eq!(sigmoid(&x), x_results);
    }

    #[test]
    fn test_softmax() {
        let x = array![[3.2, 1.3, 4.2], [0.3, 2.9, 4.0]];
        let x_results = array![
            [0.25854158, 0.0386697 , 0.7027887]
            , [0.018211273, 0.24519183, 0.73659694]];
        assert_eq!(softmax(&x), x_results);
    }

    #[test]
    fn test_cross_entropy_error() {
        let x = array![[0.3, 0.2, 0.5], [0.2, 0.1, 0.7]];
        let t = array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let x_result = 0.98305607;
        assert_eq!(cross_entropy_error(&x, &t), x_result);

        let t = array![[1.0], [2.0]];
        assert_eq!(cross_entropy_error(&x, &t), x_result);
    }
}