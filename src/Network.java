import java.util.ArrayList;

public class Network {
	private ArrayList<Neuron> inputLayer;
	private ArrayList<Neuron> hiddenLayer;
	private ArrayList<Neuron> outputLayer;
	
	private int inputSize;
	private int featureSize;
	
	public Network(int input, int feature) {
		inputSize = input;
		featureSize = feature;
		
		inputLayer = new ArrayList<>();
		hiddenLayer = new ArrayList<>();
		outputLayer = new ArrayList<>();
		
		for(int i = 0; i < inputSize; i++) {
			Neuron in = new Neuron();
			Neuron out = new Neuron();
			inputLayer.add(in);
			outputLayer.add(out);
		}
		
		for(int i = 0; i < featureSize; i++) {
			Neuron f = new Neuron();
			hiddenLayer.add(f);
		}
		
		for(Neuron from : inputLayer) {
			for(Neuron to : hiddenLayer) {
				Connection con = new Connection(to, from);
				from.output.add(con);
				to.input.add(con);
			}
		}
		
		for(Neuron from : hiddenLayer) {
			for(Neuron to : outputLayer) {
				Connection con = new Connection(to, from);
				from.output.add(con);
				to.input.add(con);
			}
		}
	}
	
	public void train(int input, int expected) {
		int[] oneHot = new int[inputSize];
		oneHot[input] = 1;
		
		double[] output = feedForward(oneHot);
	}
	
	public double[] feedForward(int[] input) {
		double[] output = new double[inputSize];
		
		for(int i = 0; i < inputLayer.size(); i++) {
			Neuron n = inputLayer.get(i);
			n.totalInput = i;
			n.setOutput();
		}
		
		for(Neuron n : hiddenLayer) {
			n.setInput();
			n.setOutput();
		}
		
		for(int i = 0; i < inputSize; i++) {
			Neuron n = outputLayer.get(i);
			n.setInput();
			n.setOutput();
			output[i] = n.totalOutput;
		}
		
		return output;
	}
}
