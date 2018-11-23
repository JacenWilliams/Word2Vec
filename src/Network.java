
public class Network {
	private int vocabSize;
	private int featureSize;
	
	private double[][][] weights;
	private double[][]   output;
	private double[]     input;

	public Network(int vocabSize, int featureSize) {
		this.vocabSize = vocabSize;
		this.featureSize = featureSize;
		
		
	}
	
	public void train(int input, int neighbor) {
		
	}
}
