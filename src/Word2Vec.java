

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class Word2Vec {
	
	private static final int SAMPLING_RATE = 10; //Determines the number of negative words trained for each training iteration
	private static int windowSize = 3;
	private static int vocabSize = 0;
	private static int featureSize = 50;
	private static HashMap<String, Word> vocab = new HashMap<>();
	private static Network network;
	private static ArrayList<File> files;
	private static int[] unigramTable;
	private static long records = 0;
	private static long counter = 0;
	private static long totalCounter = 0;
	private static double percentTotal = 0;
	
	public static void main(String[] args) {
		System.out.println(" _       ______  ____  ____ ___ _    ______________\r\n" + 
				"| |     / / __ \\/ __ \\/ __ \\__ \\ |  / / ____/ ____/\r\n" + 
				"| | /| / / / / / /_/ / / / /_/ / | / / __/ / /     \r\n" + 
				"| |/ |/ / /_/ / _, _/ /_/ / __/| |/ / /___/ /___   \r\n" + 
				"|__/|__/\\____/_/ |_/_____/____/|___/_____/\\____/   \r\n" + 
				"                                                   \r\n" + 
				""); 
		                                      
		if(args.length == 1) {
			loadTrainingData(args[0]);
			fillUnigramTable();
			train();
			listAllVectors();
			System.exit(0);
		}
		Scanner sc = new Scanner(System.in);
		String line;
		String[] items;
		System.out.print("WORD2VEC: ");

		while(!(line = sc.nextLine()).equals("exit")) {
			items = line.split("\\s+");

			if(items.length > 0) {
				if(items[0].equals("loaddata")) {
					loadTrainingData(items[1]);
					fillUnigramTable();
					System.out.println("Data loaded");
					System.out.println("Vocabulary Size: " + vocabSize);
					System.out.println("Tokens: " + records);
				} else if (items[0].equals("train")) {
					train();
					System.out.println("Training Complete");
				} else if (items[0].equals("list")) {
					listVector(items[1]);
				} else if (items[0].equals("listall")) {
					listAllVectors();
				} else if (items[0].equals("distance")) {
					if(items.length == 3) {
						cosineDistance(items[1], items[2]);
					} else {
						System.out.println("Invalid Input");
					}
				} else if(items[0].equals("closest")) {
					if(items.length == 4) {
						closestDistance(items[1], items[2], items[3]);
					} else {
						System.out.println("Invalid Input");
					}
				}
			}
			System.out.print("WORD2VEC: ");

		}
		
		sc.close();
		
	}
	
	//function to read in data from previously loaded files and train network
	private static void train() {
		System.out.println("Initializing: ");
		network = new Network(vocabSize, featureSize, unigramTable, SAMPLING_RATE);
		System.out.println("Training: ");
		
		for(File file : files) {
			trainDoc(file);
		}
		
	}
	
	private static void trainDoc(File file) {
		String line;
		String[] window = new String[(windowSize * 2) + 1];
		BufferedReader br = null;
		
		if(counter % percentTotal >= 1) {
			System.out.println("Percent Complete: " + records/totalCounter);
			counter = 0;
		}

		try {
			//load initial data window
			br = new BufferedReader(new FileReader(file));

			for(int i = 0; i < (windowSize * 2) + 1; i++) {
				window[i] = br.readLine();
			}
			
			//train initial data window
			for(int i = 0; i < windowSize; i++) {
				//printWindow(window);
				if(window[i] != null && window[windowSize] != null) {
					network.train(vocab.get(window[i]), vocab.get(window[windowSize]));
					//System.out.println("Training " + counter + " of " + records * windowSize * 2 + " | " + window[i] + " and " + window[windowSize]);
					//counter++;
					//totalCounter++;
					
				}
			}
			
			window = shiftLeft(window);
			
			while((line = br.readLine()) != null) {
				
				window[window.length - 1] = line;
				//printWindow(window);
				for(int i = 0; i < windowSize; i++) {
					if(window[i] != null && window[windowSize] != null) {
						network.train(vocab.get(window[i]), vocab.get(window[windowSize]));
						//System.out.println("Training " + counter + " of " + records * windowSize * 2 + " | " + window[i] + " and " + window[windowSize]);
						//counter++;
						//totalCounter++;
					}
				}
				
				for (int i = windowSize + 1; i < window.length; i++) {
					if(window[i] != null && window[windowSize] != null) {
						network.train(vocab.get(window[i]), vocab.get(window[windowSize]));
						//System.out.println("Training " + counter + " of " + records * windowSize * 2 + " | " + window[i] + " and " + window[windowSize]);
						//counter++;
						//totalCounter++;
					}
				}
				
				window = shiftLeft(window);
			}
			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
	
	//utility function to shift array left, last element returned as null
	private static String[] shiftLeft(String[] array) {
		if(array.length > 1) {
			for(int i = 1; i < array.length; i++) {
				array[i-1] = array[i];
			}
		}
		
		if(array.length > 0) {
			array[array.length - 1] = null;
		}
		return array;
	}
	
	//function to load data into vocabulary and determine vocabulary size
	private static void loadTrainingData(String inputdir) {
		File dir = new File(inputdir);
		files = new ArrayList<File>();
		listFiles(dir);
		counter = 0;
		totalCounter = 0;
		System.out.println("Loading Training Data: ");
		
		for( File file : files) {
			System.out.println("Loading from file: " + file);
			addData(file);
		}
		
		vocabSize = vocab.size();
		percentTotal = records / 100;
	}
	
	//utility function to add line of data to vocabulary
	private static void addData(File file) {
		try {
			String line;
			int index = 0;
			
			BufferedReader br = new BufferedReader(new FileReader(file));
			
			while((line = br.readLine()) != null) {
				records++;
				if(!vocab.containsKey(line)) {
					vocab.put(line, new Word(line, index));
					index++;
				} else {
					vocab.get(line).count++;
				}
			}
			
			br.close();
			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
	
	//utility function to expand directory and add all files contained to files arraylist
	private static void listFiles(File dir) {		
		File[] fList = dir.listFiles();
		if(fList != null) {
			for (File file : fList) {
				if (file.isFile()) {
					files.add(file);
				} else if (file.isDirectory()) {
					listFiles(file);
				}
			}
		}
	}
	
	
	//*method to fill unigramTable*
	//unigramTable is used to quickly select random words for Negative Sampling.
	//unigramTable is initialized to 100000000 and each word is added to the table
	//according to its weight. Weight is calculated as w^3/4. The table then stores
	//the index of the word.
	private static void fillUnigramTable() {
		int count = 0;
		int weight = 0;
		unigramTable = new int[100000000];
		
		for(Word word : vocab.values()) {
			weight = (int) Math.pow(word.count, (3/4));

			for(int i = 0; i < weight; i++) {
				if (count < 100000000) {
					unigramTable[count] = word.index;
					count++;
				}
			}
		}
	}
	
	private static void listVector(String word) {
		System.out.println("Vector for " + word + " : ");
		
		if(!vocab.containsKey(word)) {
			System.out.println("Error: Word not in vocabulary");
			return;
		}
		
		Word key = vocab.get(word);
		double[] vector = network.getVector(key);
		Matrix.printVector(vector);
	}
	
	private static void listAllVectors() {
		for(Word word : vocab.values()) {
			listVector(word.value);
		}
	}
	
	private static void cosineDistance(String a, String b) {
		if(!vocab.containsKey(a)) {
			System.out.println("Error: " + a + " not in vocabulary");
			return;
		}
		
		if(!vocab.containsKey(b)) {
			System.out.println("Error: " + b + " not in vocabulary");
			return;
		}
		
		System.out.println(Matrix.cosineDistance(network.getVector(vocab.get(a)), network.getVector(vocab.get(b))));
		
		
	}
	
	private static void closestDistance(String a, String b, String c) {
		if(!vocab.containsKey(a)) {
			System.out.println("Error: " + a + " not in vocabulary");
			return;
		}
		
		if(!vocab.containsKey(b)) {
			System.out.println("Error: " + b + " not in vocabulary");
			return;
		}
		
		if(!vocab.containsKey(c)) {
			System.out.println("Error: " + b + " not in vocabulary");
			return;
		}
		
		Word word1 = vocab.get(a);
		Word word2 = vocab.get(b);
		Word word3 = vocab.get(c);
		
		double[] aVec = network.getVector(word1);
		double[] bVec = network.getVector(word2);
		double[] cVec = network.getVector(word3);
		
		double[] comp = Matrix.subtract(aVec, bVec);
		comp = Matrix.add(comp, cVec);
		
		Word minWord = null;
		double minValue = Double.MAX_VALUE;
		
		for(Word word : vocab.values()) {
			double[] wordVec = network.getVector(word);
			if(Matrix.cosineDistance(comp, wordVec) < minValue && word.index != word1.index
					&& word.index != word2.index && word.index != word3.index) {
				minValue = Matrix.cosineDistance(comp, wordVec);
				minWord = word;
			}
				
		}
		
		System.out.println(a + " : " + b + " | " + c + " : " + minWord.value);
	}
	
}