import java.util.ArrayList;
import java.util.HashMap;

public class Neuron {
	public ArrayList<Connection> input;
	public HashMap<Integer, Connection> directedOutput;
	//public ArrayList<Connection> output;
	public double totalInput = 0;
	public double totalOutput = 0;
	public double error = 0;
	
	public Neuron() {
		input = new ArrayList<>();
		//output = new ArrayList<>();
		directedOutput = new HashMap<>();
	}
	
	//calculates sigmoid activation function and assigns to totalOutput;
	public void setOutput() {
		totalOutput = 1/(1+(Math.exp(-totalInput)));
	}
	
	
	//returns weighted total of all inputs
	public void setInput() {
		double sum = 0;
		for(Connection con : input) {
			sum += con.calculateInput();
		}
		totalInput = sum;
	}
	
	public void addOutput(Connection con, int index) {
		//output.add(con);

		directedOutput.put(index, con);

		
	}
	
	public double derivative() {
		return totalOutput * (1 - totalOutput);
	}
	
}
