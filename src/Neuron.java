import java.util.ArrayList;

public class Neuron {
	public ArrayList<Connection> input;
	public ArrayList<Connection> output;
	public double totalInput = 0;
	public double totalOutput = 0;
	
	public Neuron() {
		this.input = new ArrayList<>();
		this.output = new ArrayList<>();
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
	
}
