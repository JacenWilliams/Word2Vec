import java.util.ArrayList;
import java.util.HashMap;

public class Network {
	private ArrayList<Neuron> inputLayer;
	private ArrayList<Neuron> hiddenLayer;
	private ArrayList<Neuron> outputLayer;
	
	private HashMap<String, Word> vocab;
	private int[] unigramTable;
	private int samplingRate;
	
	private int inputSize;
	private int featureSize;
	private final double ALPHA = 0.2;
	
	public Network(int input, int feature, HashMap<String, Word> vocab, int[] unigramTable, int samplingRate) {
		inputSize = input;
		featureSize = feature;
		
		this.vocab = vocab;
		this.unigramTable = unigramTable;
		this.samplingRate = samplingRate;
		
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
		
		for(int i = 0; i < inputLayer.size(); i++) { 
			Neuron from = inputLayer.get(i);
			for(int j = 0; j < hiddenLayer.size(); j++) { 
				Neuron to = hiddenLayer.get(j);
				Connection con = new Connection(to, from);
				from.addOutput(con, j);
				to.input.add(con);
			}
		}
		
		for(int i = 0; i < hiddenLayer.size(); i++) { 
			Neuron from = hiddenLayer.get(i);
			for(int j = 0; j < outputLayer.size(); j++) { 
				Neuron to = outputLayer.get(j);
				Connection con = new Connection(to, from);
				from.addOutput(con, j);
				to.input.add(con);
			}
		}
	}
	
	public void train(Word input, Word expected) {
		int[] oneHot = new int[inputSize];
		oneHot[input.index] = 1;
		
		double[] rawOutput = feedForward(oneHot);
		double[] output = softmax(rawOutput);
		backPropagate(expected, output, inputLayer.get(input.index));
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
	
	//converts sigmoid output array into probability distribution using softmax function
	public double[] softmax(double[] input) {
		double[] output = new double[input.length];
		double eSum = 0;
		
		for(int i = 0; i < input.length; i++) {
			eSum += Math.exp(input[i]);
		}
		
		for(int i = 0; i < input.length; i++) {
			output[i] = Math.exp(input[i]) / eSum;
		}
		
		return output;
	}
	
	public void backPropagate(Word expected, double[] output, Neuron input) {
		ArrayList<Integer> samples = sample(expected.index);
		double delta = 0;
		
		//calculate weighted error for output layer
		for(int sample : samples) {
			Neuron out = outputLayer.get(sample);
			double target = 0;
			if (sample == expected.index) {
				target = 1;
			}
			
			//weighted error is calculated as absolute error * derivative of sigmoid function
			out.error = specificError(target, output[sample]) * out.derivative();
			
		}
		
		//calculate weighted error for hidden layer
		for(int i = 0; i < hiddenLayer.size(); i++) {
			Neuron hidden = hiddenLayer.get(i);
			double sum = 0;
			for(int sample : samples) {
				Connection con = hidden.directedOutput.get(sample);
				sum += con.weight * con.to.error;
			}
			
			hidden.error = sum * hidden.derivative();
		}
		
		//update weights for hidden layer to output layer
		for(int sample : samples) {
			Neuron out = outputLayer.get(sample);
			for(Connection con : out.input) {
				con.weight = con.weight - (ALPHA * con.from.totalOutput * out.error);
			}
		}
		
		//update weights for input layer to hidden layer
		for(Connection con : input.directedOutput.values()) {
			con.weight = con.weight - (ALPHA * con.from.totalOutput * con.to.error);
		}
		
	}
	
	//calculate squared error for single output
	public double specificError(double expected, double output) {
		//return .5 * Math.pow(expected - output, 2);
		return expected - output;
	}
	
	//calculate total squared sum error for all outputs
	public double totalError(Word expected, double[] output) {
		double total = 0;
		double error = 0;
		int value = 0;
		
		for(int i = 0; i < output.length; i++) {
			if(expected.index == i) {
				value = 1;
			} else {
				value = 0;
			}
			
			error = (1/2) * (Math.pow(value - output[i], 2));
			total += error;
		}
		
		return total;
	}
	
	//returns an arraylist containing indexes of randomly selected words using unigramTable
	//correct index is supplied to ensure it is not selected;
	public ArrayList<Integer> sample(int correctIndex) {
		ArrayList<Integer> samples = new ArrayList<Integer>();
		int index = 0;
		
		for(int i = 0; i < samplingRate; i++) {
			index =(int) Math.random() * 100000000;
			
			if(unigramTable[index] == correctIndex) {
				i--;
			} else {
				samples.add(unigramTable[index]);
			}
		}
		
		return samples;
	}
	
	public double[] getVector(Word input) {
		double[] vec = new double[featureSize];
		Neuron n = inputLayer.get(input.index);
		ArrayList<Connection> weights = (ArrayList<Connection>) n.directedOutput.values();
		
		for(int i = 0; i < featureSize; i++) {
			Connection con = weights.get(i);
			vec[i] = con.weight;
		}
		
		return vec;
	}
}
