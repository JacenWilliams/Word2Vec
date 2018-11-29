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
	
	public static double[] subtract(double[] a, double[] b ) {
		if(a.length != b.length) {
			System.out.println("Error: Invalid matrix operation");
			return null;
		}
		
		double[] c = new double[a.length];
		
		for(int i = 0; i < a.length; i++) {
			c[i] = a[i] - b[i];
		}
		
		return c;
	}
	
	public static double[] add(double[] a, double[] b ) {
		if(a.length != b.length) {
			System.out.println("Error: Invalid matrix operation");
			return null;
		}
		
		double[] c = new double[a.length];
		
		for(int i = 0; i < a.length; i++) {
			c[i] = a[i] + b[i];
		}
		
		return c;
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
	
	public static void printVector(double[] a) {
		for(int i = 0; i < a.length; i++) {
			System.out.print("[" + a[i] + "] ");
		}
		System.out.print("\n");
	}
}
