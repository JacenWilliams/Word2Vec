public final class Matrix {
	
	private Matrix() {
		
	}
	
	public static double dot(double[] a, double[] b) {
		double sum = 0;
		
		if(a.length != b.length) {
			System.out.println("Error: Invalid dot multiplication");
			return -1;

		}
		
		for(int i = 0; i < a.length; i++) {
			sum += a[i] * b[i];
		}
		
		return sum;
	}
	
	public static double cosineDistance(double[] a, double[] b) {
		double dot = dot(a, b);
		double x = 0;
		double y = 0;
		
		for(int i = 0; i < a.length; i++) {
			x += Math.pow(a[i], 2);
			y += Math.pow(b[i], 2);
		}
		
		return 1 - (dot / ((Math.sqrt(x)) * (Math.sqrt(y))));
	}
}
