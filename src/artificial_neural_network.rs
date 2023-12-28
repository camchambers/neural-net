use std::fmt;
use crate::neuron::Neuron;

pub struct ArtificialNeuralNetwork {
    layers: Vec<usize>,
    bias: f64,
    eta: f64,
    values: Vec<Vec<f64>>,
    d: Vec<Vec<f64>>,
    network: Vec<Vec<Neuron>>,
}

impl ArtificialNeuralNetwork {
    pub fn new(layers: Vec<usize>, bias: f64, eta: f64) -> ArtificialNeuralNetwork {
        let mut values = Vec::new();
        let mut d = Vec::new();
        let mut network = Vec::new();

        for i in 0..layers.len() {
            values.push(vec![0.0; layers[i]]);
            d.push(vec![0.0; layers[i]]);
            network.push(Vec::new());

            if i > 0 {
                for _ in 0..layers[i] {
                    network[i].push(Neuron::new(layers[i - 1], bias));
                }
            }
        }

        ArtificialNeuralNetwork {
            layers,
            bias,
            eta,
            values,
            d,
            network,
        }
    }

    pub fn run(&mut self, x: Vec<f64>) -> Vec<f64> {
        self.values[0] = x;
        for i in 1..self.network.len() {
            for j in 0..self.layers[i] {
                self.values[i][j] = self.network[i][j].run(self.values[i - 1].clone());
            }
        }
        self.values.last().unwrap().clone()
    }

    pub fn train(&mut self, samples: Vec<Vec<f64>>, labels: Vec<Vec<f64>>, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_mse = 0.0;
            for i in 0..samples.len() {
                self.run(samples[i].clone());
                total_mse += self.backpropagate(samples[i].clone(), labels[i].clone());
            }
            total_mse /= samples.len() as f64;
            if epoch % 1000 == 0 {
                println!("Epoch: {}, MSE: {}", epoch, total_mse);
            }
        }
    }

    pub fn test(&mut self, samples: Vec<Vec<f64>>, labels: Vec<Vec<f64>>) {
        let mut total_mse = 0.0;
        for i in 0..samples.len() {
            let output = self.run(samples[i].clone());

            println!("Input: {:?}", samples[i]);
            println!("Output: {:?}", output);

            let mut mse = 0.0;
            for j in 0..labels[i].len() {
                let error = labels[i][j] - output[j];
                mse += error * error;
            }
            mse /= labels[i].len() as f64;
            total_mse += mse;
        }

        total_mse /= samples.len() as f64;
        println!("Test MSE: {}", total_mse);
    }

    pub fn backpropagate(&mut self, x: Vec<f64>, y: Vec<f64>) -> f64 {
        // Run the input through the network.
        let outputs = self.run(x.clone());

        // Calculate the mean squared error.
        let mut error = Vec::new();
        let mut mse = 0.0;
        for i in 0..y.len() {
            error.push(y[i] - outputs[i]);
            mse += error[i] * error[i];
        }
        mse /= *self.layers.last().unwrap_or(&0) as f64;
        
        // Calculate the error term of each output neuron.
        for i in 0..outputs.len() {
            self.d.last_mut().unwrap()[i] = outputs[i] * (1.0 - outputs[i]) * error[i];
        }

        // Calculate the error term of each neuron in each hidden layer.
        for i in (1..self.network.len()).rev() {
            for h in 0..self.network[i].len() {
                let mut fwd_error = 0.0;
                for k in 0..self.layers[i] {
                    fwd_error += self.network[i][k].weights[h] * self.d[i][k];
                }
                self.d[i - 1][h] =
                    self.values[i - 1][h] * (1.0 - self.values[i - 1][h]) * fwd_error;
            }
        }

        // Calculate the deltas and update the weights for each neuron.
        for i in 1..self.network.len() {
            for j in 0..self.layers[i] {
                for k in 0..self.layers[i - 1] + 1 {
                    let delta;
                    if k == self.layers[i - 1] {
                        // The delta is equal to the learning rate multiplied by the error term, multiplied by the bias.
                        delta = self.eta * self.d[i][j] * self.bias;
                    } else {
                        delta = self.eta * self.d[i][j] * self.values[i - 1][k];
                    }
                    self.network[i][j].weights[k] += delta;
                }
            }
        }

        mse
    }

}

impl fmt::Display for ArtificialNeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Printing Neural Network:\n")
    }
}
