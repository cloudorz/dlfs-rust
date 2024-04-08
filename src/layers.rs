use ndarray::{Array, Axis, Ix1, Ix2};
use crate::functions::*;

type NNMatrix = Array<f32, Ix2>;

#[derive(Debug)]
pub struct Relu {
    #[allow(dead_code)]
    mask: Option<Array<bool, Ix2>>,
}

impl Relu  {
    pub fn new() -> Self {
        Self { mask: None }
    }
}

impl Relu  {
    pub fn forward(&mut self, x:   NNMatrix) ->   NNMatrix {
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

    pub fn backward(&self, d_out: &  NNMatrix) ->   NNMatrix {
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
struct Sigmoid {
    #[allow(dead_code)]
    out: Option<NNMatrix>
}

impl Sigmoid  {
    pub fn new() -> Self {
        Self {
            out: None
        }
    }
}

impl Sigmoid  {
    pub fn forward(&mut self, x: &  NNMatrix) -> NNMatrix  {
        let out = sigmoid(x);
        self.out = Some(out.clone());
        out
    }

    pub fn backward(&self, d_out: &  NNMatrix) -> NNMatrix  {
        let out = self.out.as_ref().unwrap();
        out.shape();

        d_out * (1.0 - out) * out
    }
}


struct Affine {
    bias:   NNMatrix,
    d_bias: Option<Array<f32, Ix1>>,
    weight:   NNMatrix,
    d_weight: Option<  NNMatrix>,
    x: Option<  NNMatrix>,
}

impl Affine {
    pub fn new(weight:   NNMatrix, bias:   NNMatrix) -> Self {
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
    pub fn forward(&mut self, x: &  NNMatrix) ->   NNMatrix {
        self.x = Some(x.clone());
        let out = x.dot(&self.weight) + &self.bias;

        out
    }

    pub fn backward(&mut self, d_out: &  NNMatrix) ->   NNMatrix {
        let d_x = d_out.dot(&self.weight.t());
        self.d_weight = Some(self.x.as_ref().unwrap().t().dot(d_out));
        self.d_bias = Some(d_out.sum_axis(Axis(0)));

        d_x
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array};

    #[test]
    fn test_relu_forward() {
        let mut relu_layer = Relu::new();
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
        let mut sigmoid_layer = Sigmoid::new();
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