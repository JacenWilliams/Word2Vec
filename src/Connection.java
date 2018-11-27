
public class Connection {
	public Neuron to;
	public Neuron from;
	public double weight;
	
	public Connection(Neuron to, Neuron from, double weight) {
		this.to = to;
		this.from = from;
		this.weight = weight;
	}
	
	public Connection(Neuron to, Neuron from) {
		this.to = to;
		this.from = from;
		this.weight = Math.random();
	}
	
	public double calculateInput() {
		return from.totalOutput * weight;
	}
}
