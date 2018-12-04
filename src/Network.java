import java.util.ArrayList;

public class Network {
	private ArrayList<Neuron> inputLayer;
	private ArrayList<Neuron> hiddenLayer;
	private ArrayList<Neuron> outputLayer;
	
	private int[] unigramTable;
	private int samplingRate;
	
	private int inputSize;
	private int featureSize;
	private final double ALPHA = 0.025;
	
	public Network(int input, int feature, int[] unigramTable, int samplingRate) {
		inputSize = input;
		featureSize = feature;
		
		this.unigramTable = unigramTable;
		this.samplingRate = samplingRate;
		
		inputLayer = new ArrayList<>();
		hiddenLayer = new ArrayList<>();
		outputLayer = new ArrayList<>();
		
		//initialize input and output layer neurons
		for(int i = 0; i < inputSize; i++) {
			Neuron in = new Neuron();
			Neuron out = new Neuron();
			inputLayer.add(in);
			outputLayer.add(out);
		}
		
		//initialize hidden layer neurons
		for(int i = 0; i < featureSize; i++) {
			Neuron f = new Neuron();
			hiddenLayer.add(f);
		}
		
		//initialize input to hidden layer connections
		for(int i = 0; i < inputLayer.size(); i++) { 
			Neuron from = inputLayer.get(i);
			for(int j = 0; j < hiddenLayer.size(); j++) { 
				Neuron to = hiddenLayer.get(j);
				Connection con = new Connection(to, from);
				from.addOutput(con, j);
				to.input.add(con);
			}
		}
		
		//initialize hidden to output layer connections
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
	
	public void train(Word input, Word expected)  {
		
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
			n.totalInput = input[i];
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
		//System.out.println("Sampling...");
		ArrayList<Integer> samples = sample(expected.index);
		
		//System.out.println("Calculating output layer error...");
		//calculate error for output layer neurons
		for(int sample : samples) {
			Neuron out = outputLayer.get(sample);
			double target = 0;
			if (sample == expected.index) {
				target = 1;
			}
			
			
			out.error = specificError(target, output[sample]);
			
		}
		
		//System.out.println("Calculating hidden layer error...");

		//calculate error for hidden layer
		for(int i = 0; i < hiddenLayer.size(); i++) {
			Neuron hidden = hiddenLayer.get(i);
			double sum = 0;
			for(int sample : samples) {
				Connection con = hidden.directedOutput.get(sample);
				sum += con.weight * con.to.error;
			}
			
			hidden.error = sum;
		}
		
		//System.out.println("Calculating output layer weights...");

		//update weights for hidden layer to output layer
		for(int sample : samples) {
			Neuron out = outputLayer.get(sample);
			for(Connection con : out.input) {
				con.weight = con.weight - (ALPHA * con.from.totalOutput * out.error);
			}
		}
		
		//System.out.println("Calculating hidden layer weights...");

		//update weights for input layer to hidden layer
		for(Connection con : input.directedOutput.values()) {
			con.weight = con.weight - (ALPHA * con.from.totalOutput * con.to.error);
		}
		
	}
	
	//calculate error for single output
	public double specificError(double expected, double output) {
		//return .5 * Math.pow(expected - output, 2);
		return output - expected;
	}
	

	
	//returns an arraylist containing indexes of randomly selected words using unigramTable
	//correct index is supplied to ensure it is not selected;
	public ArrayList<Integer> sample(int correctIndex) {
		ArrayList<Integer> samples = new ArrayList<Integer>();
		int index = 0;
		//System.out.println(samples.size() + " : " + samplingRate);
		for(int i = 0; i < samplingRate; i++) {
			index =(int) Math.random() * 100000000;
			//System.out.println(i + " : " + index);
			samples.add(unigramTable[index]);
		}
		
		return samples;
	}
	
	
	//converts weights from input layer to hidden to vector for specific input
	public double[] getVector(Word input) {
		double[] vec = new double[featureSize];
		Neuron n = inputLayer.get(input.index);
		ArrayList<Connection> weights = new ArrayList<>(n.directedOutput.values());
		
		for(int i = 0; i < featureSize; i++) {
			Connection con = weights.get(i);
			vec[i] = con.weight;
		}
		
		return vec;
	}
	
	//converts weights from input layer to hidden layer to array of vectors for al inputs
	public double[][] getVectors() {
		double[][] vectors = new double[inputSize][featureSize];
		Neuron n;
		
		for(int i = 0; i < inputSize; i++) {
			n = inputLayer.get(i);
			ArrayList<Connection> weights = new ArrayList<>(n.directedOutput.values());
			for (int j = 0; j < featureSize; j++) {
				Connection con = weights.get(j);
				vectors[i][j] = con.weight;
			}
		}
		
		return vectors;
	}
}
