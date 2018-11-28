

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

public class Word2Vec {
	
	private static final int SAMPLING_RATE = 10; //Determines the number of negative words trained for each training cycle
	private static int windowSize = 5;
	private static int vocabSize = 0;
	private static int featureSize = 300;
	private static HashMap<String, Word> vocab = new HashMap<>();
	private static Network network;
	private static ArrayList<File> files;
	private static int[] unigramTable;
	
	public static void main(String[] args) {

		Scanner sc = new Scanner(System.in);
		String line;
		String[] items;
		
		while(!(line = sc.nextLine()).equals("exit")) {
			items = line.split("\\s+");
			
			if(items.length > 0) {
				if(items[0].equals("loaddata")) {
					loadTrainingData(items[1]);
					fillUnigramTable();
				} else if (items[0].equals("train")) {
					train();
				}
			}
		}
		
	}
	
	//function to read in data from previously loaded files and train network
	private static void train() {
		int docs = files.size();
		String line;
		String[] window = new String[(windowSize * 2) + 1];
		BufferedReader br = null;
		network = new Network(vocabSize, featureSize, vocab, unigramTable, SAMPLING_RATE);
		
		try {
			//load initial data window
			for(int i = 0; i < (windowSize * 2) + 1; i++) {
				window[i] = loadLine(br);
			}
			
			//train initial data window
			for(int i = 0; i < windowSize; i++) {
				if(window[i] != null && window[windowSize] != null) {
					network.train(vocab.get(window[i]), vocab.get(window[windowSize]));
				}
			}
			
			window = shiftLeft(window);
			
			while((line = loadLine(br)) != null) {
				window[windowSize - 1] = line;
				
				for(int i = 0; i < windowSize; i++) {
					if(window[i] != null && window[windowSize] != null) {
						network.train(vocab.get(window[i]), vocab.get(window[windowSize]));
					}
				}
				
				for (int i = windowSize + 1; i < window.length; i++) {
					if(window[i] != null && window[windowSize] != null) {
						network.train(vocab.get(window[i]), vocab.get(window[windowSize]));
					}
				}
				
				window = shiftLeft(window);
			}
			
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
	
	//utility function to load next line in document or first line of next document
	private static String loadLine(BufferedReader br) throws IOException {
		String line;
		
		if(br == null) {
			br = new BufferedReader( new FileReader(files.get(0)));
			files.remove(0);
		}
		
		while((line = br.readLine()) == null) {
			if(!files.isEmpty()) {
				br = new BufferedReader( new FileReader(files.get(0)));
				files.remove(0);
			} else {
				return null;
			}
		}
		
		return line;
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
		
		for( File file : files) {
			addData(file);
		}
		
		vocabSize = vocab.size();
	}
	
	//utility function to add line of data to vocabulary
	private static void addData(File file) {
		try {
			String line;
			int index = 0;
			
			BufferedReader br = new BufferedReader(new FileReader(file));
			
			while((line = br.readLine()) != null) {
				if(!vocab.containsKey(line)) {
					vocab.put(line, new Word(line, index));
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
		ArrayList<File> subdir;
		
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
}

	
