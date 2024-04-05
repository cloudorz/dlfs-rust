use ndarray::{Array, Dimension};

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

    pub fn backward(&self, dout: &Array<f32, D>) -> Array<f32, D> {
        let mut clone_dout = dout.clone();
        if let Some(ref _mask) = self.mask {
            clone_dout.zip_mut_with(_mask,
                 |x_value, bool_value| {
                     if *bool_value { *x_value = 0.0 }; 
                });
        }
        clone_dout
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
        let dout_vec = array![[0.3, -0.5, -2.8], [0.1, 0.4, -1.2]];
        let result_vec = array![[0.3, 0.0, -2.8], [0.0, 0.4, 0.0]];
        assert_eq!(relu_layer.backward(&dout_vec), result_vec);
    }
}