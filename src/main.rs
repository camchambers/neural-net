mod neuron;
mod artificial_neural_network;
use artificial_neural_network::ArtificialNeuralNetwork;

fn and_example() {
    println!("AND Logic Gate Example");

    let mut ann = ArtificialNeuralNetwork::new(vec![2, 2, 1], 1.0, 0.5);

    // AND training data.
    let training_samples = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let training_labels = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];

    // Train the network on the AND logic gate.
    ann.train(training_samples, training_labels, 10000);

    // Test the network on the AND logic gate.
    let testing_samples = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let testing_labels = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];
    ann.test(testing_samples, testing_labels);

    println!();
}

fn main() {
    and_example();
}
