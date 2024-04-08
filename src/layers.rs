use ndarray::{Array, array, Axis, Dimension, Ix1, Ix2};
use crate::functions::*;

type NNMatrix<D> = Array<f32, D>;

#[derive(Debug)]
pub struct Relu<D: Dimension> {
    #[allow(dead_code)]
    mask: Option<Array<bool, D>>,
}

impl<D: Dimension> Relu<D> {
    pub fn new() -> Self {
        Self { mask: None }
    }
}

impl<D: Dimension> Relu<D> {
    pub fn forward(&mut self, x: Array<f32, D>) -> Array<f32, D> {
        self.mask = Some(x.map(|x| *x < 0.0));

        if let Some(ref _mask) = self.mask  {
            let mut clone_x = x.clone();
            clone_x.zip_mut_with(_mask,
                 |x_value, bool_value| { 
                    if *bool_value { *x_value = 0.0 };
                });
            clone_x
        } else {
            x
        }
    }

    pub fn backward(&self, d_out: &Array<f32, D>) -> Array<f32, D> {
        let mut clone_d_out = d_out.clone();
        if let Some(ref _mask) = self.mask {
            clone_d_out.zip_mut_with(_mask,
                 |x_value, bool_value| {
                     if *bool_value { *x_value = 0.0 }; 
                });
        }
        clone_d_out
    }
    
}


#[derive(Debug)]
struct Sigmoid<D: Dimension> {
    #[allow(dead_code)]
    out: Option<Array<f32, D>>
}

impl<D: Dimension> Sigmoid<D> {
    pub fn new() -> Self {
        Self {
            out: None
        }
    }
}

impl<D: Dimension> Sigmoid<D> {
    pub fn forward(&mut self, x: &NNMatrix<D>) -> NNMatrix<D> {
        let out = sigmoid(x);
        self.out = Some(out.clone());
        out
    }

    pub fn backward(&self, d_out: &NNMatrix<D>) -> NNMatrix<D> {
        let out = self.out.as_ref().unwrap();
        out.shape();

        d_out * (1.0 - out) * out
    }
}

type NN2Matrix = Array<f32, Ix2>;

struct Affine {
    bias: NN2Matrix,
    d_bias: Option<Array<f32, Ix1>>,
    weight: NN2Matrix,
    d_weight: Option<NN2Matrix>,
    x: Option<NN2Matrix>,
}

impl Affine {
    pub fn new(weight: NN2Matrix, bias: NN2Matrix) -> Self {
        Self {
            bias,
            d_bias: None,
            weight,
            d_weight: None,
            x: None,
        }
    }
}

impl Affine {
    pub fn forward(&mut self, x: &NN2Matrix) -> NN2Matrix {
        self.x = Some(x.clone());
        let out = x.dot(&self.weight) + &self.bias;

        out
    }

    pub fn backward(&mut self, d_out: &NN2Matrix) -> NN2Matrix {
        let d_x = d_out.dot(&self.weight.t());
        self.d_weight = Some(self.x.as_ref().unwrap().t().dot(d_out));
        self.d_bias = Some(d_out.sum_axis(Axis(0)));

        d_x
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Ix2};

    #[test]
    fn test_relu_forward() {
        let mut relu_layer = Relu::<Ix2>::new();
        let x_vec = array![
            [-1.0, 0.0, 1.0f32],
            [100.0, -10.0, 0.0],
            ];
        let result_vec = array![
            [0.0, 0.0, 1.0f32],
            [100.0, 0.0, 0.0],
            ];
        assert_eq!(relu_layer.forward(x_vec), result_vec);
    }

    #[test]
    fn test_relu_backward() {
        let relu_layer = Relu { mask: Some(array![[false, true, false], [true, false, true]]) };
        let d_out_vec = array![[0.3, -0.5, -2.8], [0.1, 0.4, -1.2]];
        let result_vec = array![[0.3, 0.0, -2.8], [0.0, 0.4, 0.0]];
        assert_eq!(relu_layer.backward(&d_out_vec), result_vec);
    }

    #[test]
    fn test_sigmoid_forward() {
        // a bit of silly case?
        let mut sigmoid_layer = Sigmoid::<Ix2>::new();
        let x_vec = array![
            [18.0, 0.0, -100.0f32],
            [-100.0, 50.0, 0.0],
            ];
        let result_vec = array![
            [1.0, 0.5f32, 0.0],
            [0.0, 1.0, 0.5],
            ];
        assert_eq!(sigmoid_layer.forward(&x_vec), result_vec);
    }

    #[test]
    fn test_sigmoid_backward() {
        // TODO: use the regular way of calculating differentials to check the result ?
    }
}