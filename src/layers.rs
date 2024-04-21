use crate::functions::*;
use crate::types::*;
use ndarray::{Array2, Axis};

#[derive(Debug)]
pub struct Relu {
    #[allow(dead_code)]
    mask: Option<Array2<bool>>,
}

impl Relu {
    pub fn new() -> Self {
        Self { mask: None }
    }
}

impl Relu {
    pub fn forward(&mut self, x: &NNMatrix) -> NNMatrix {
        self.mask = Some(x.mapv(|x| x < 0.0));

        let mut clone_x = x.clone();
        clone_x.zip_mut_with(self.mask.as_ref().unwrap(), |x_value, bool_value| {
            if *bool_value {
                *x_value = 0.0
            };
        });
        clone_x
    }

    pub fn backward(&self, d_out: &NNMatrix) -> NNMatrix {
        let mut clone_d_out = d_out.clone();
        clone_d_out.zip_mut_with(self.mask.as_ref().unwrap(), |x_value, bool_value| {
            if *bool_value {
                *x_value = 0.0
            };
        });
        clone_d_out
    }
}

#[derive(Debug)]
pub struct Sigmoid {
    #[allow(dead_code)]
    out: Option<NNMatrix>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { out: None }
    }
}

impl Sigmoid {
    pub fn forward(&mut self, x: &NNMatrix) -> NNMatrix {
        let out = sigmoid(x);
        self.out = Some(out.clone());
        out
    }

    pub fn backward(&self, d_out: &NNMatrix) -> NNMatrix {
        let out = self.out.as_ref().unwrap();

        d_out * (1.0 - out) * out
    }
}

#[derive(Debug)]
pub struct Affine {
    bias: NNBiasType,
    d_bias: Option<NNBiasType>,
    weight: NNMatrix,
    d_weight: Option<NNMatrix>,
    x: Option<NNMatrix>,
}

impl Affine {
    pub fn new(weight: NNMatrix, bias: NNBiasType) -> Self {
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
    pub fn forward(&mut self, x: &NNMatrix) -> NNMatrix {
        self.x = Some(x.clone());

        x.dot(&self.weight) + &self.bias
    }

    pub fn backward(&mut self, d_out: &NNMatrix) -> NNMatrix {
        let d_x = d_out.dot(&self.weight.t());
        self.d_weight = Some(self.x.as_ref().unwrap().t().dot(d_out));
        self.d_bias = Some(d_out.sum_axis(Axis(0)));

        d_x
    }

    pub fn update(&mut self, learning_rate: NNFloat) {
        self.weight = &self.weight - self.d_weight.as_ref().unwrap() * learning_rate;
        self.bias = &self.bias - self.d_bias.as_ref().unwrap() * learning_rate;
    }
}

#[derive(Debug)]
pub struct SoftmaxWithLoss {
    loss: Option<NNFloat>,
    y: Option<NNMatrix>,
    t: Option<NNMatrix>,
}

impl SoftmaxWithLoss {
    pub fn new() -> Self {
        Self {
            loss: None,
            y: None,
            t: None,
        }
    }
}

impl SoftmaxWithLoss {
    pub fn forward(&mut self, x: &NNMatrix, t: &NNMatrix) -> NNFloat {
        self.t = Some(t.clone());
        self.y = Some(softmax(x));
        self.loss = Some(cross_entropy_error(
            self.y.as_ref().unwrap(),
            self.t.as_ref().unwrap(),
        ));

        self.loss.unwrap()
    }

    pub fn backward(&self) -> NNMatrix {
        let ref_t = self.t.as_ref().unwrap();
        let ref_y = self.y.as_ref().unwrap();
        let batch_size = ref_t.shape()[0] as NNFloat;

        if ref_y.len() == ref_t.len() {
            (ref_y - ref_t) / batch_size
        } else {
            let mut dx = ref_y.clone();
            for (i, arr) in ref_t.axis_iter(Axis(0)).enumerate() {
                for value in arr.iter() {
                    dx[[i, *value as usize]] -= 1.0;
                }
            }

            dx / batch_size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_relu_forward() {
        let mut relu_layer = Relu::new();
        let x_vec = array![[-1.0, 0.0, 1.0], [100.0, -10.0, 0.0],];
        let result_vec = array![[0.0, 0.0, 1.0], [100.0, 0.0, 0.0],];
        assert_eq!(relu_layer.forward(&x_vec), result_vec);
    }

    #[test]
    fn test_relu_backward() {
        let relu_layer = Relu {
            mask: Some(array![[false, true, false], [true, false, true]]),
        };
        let d_out_vec = array![[0.3, -0.5, -2.8], [0.1, 0.4, -1.2]];
        let result_vec = array![[0.3, 0.0, -2.8], [0.0, 0.4, 0.0]];
        assert_eq!(relu_layer.backward(&d_out_vec), result_vec);
    }

    #[test]
    fn test_sigmoid_forward() {
        // a bit of silly case?
        let mut sigmoid_layer = Sigmoid::new();
        let x_vec = array![[18.0, 0.0, -100.0], [-100.0, 50.0, 0.0],];
        let result_vec = array![[1.0, 0.5, 0.0], [0.0, 1.0, 0.5],];
        assert_eq!(sigmoid_layer.forward(&x_vec), result_vec);
    }

    #[test]
    fn test_sigmoid_backward() {
        // TODO: use the regular way of calculating differentials to check the result ?
    }
}
