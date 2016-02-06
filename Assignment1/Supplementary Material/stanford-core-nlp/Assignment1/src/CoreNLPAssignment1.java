
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.DocumentPreprocessor;

import java.util.*;
import java.util.Map.Entry;
import java.io.*;

public class CoreNLPAssignment1 {
	
	/**
	 * 
	 * @param unsortMap : Unsorted map of word and their frequency  
	 * @param order : false for descending order
	 * @return : returns sorted map according to value
	 */
	private static Map<String, Integer> sortByComparator(Map<String, Integer> unsortMap, final boolean order)
    {

        List<Entry<String, Integer>> list = new LinkedList<Entry<String, Integer>>(unsortMap.entrySet());

        // Sorting the list based on values
        Collections.sort(list, new Comparator<Entry<String, Integer>>()
        {
            public int compare(Entry<String, Integer> o1,
                    Entry<String, Integer> o2)
            {
                if (order)
                {
                    return o1.getValue().compareTo(o2.getValue());
                }
                else
                {
                    return o2.getValue().compareTo(o1.getValue());

                }
            }
        });

        // Maintaining insertion order with the help of LinkedList
        Map<String, Integer> sortedMap = new LinkedHashMap<String, Integer>();
        for (Entry<String, Integer> entry : list)
        {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        return sortedMap;
    }

	/**
	 * 
	 * @param fileName : name of input file in string format with complete path default path is \path\to\workspace\PorjectDir\
	 * @return returns all the content of input file in String format
	 * @throws IOException : checks for IO error
	 */
    private static String readFile(String fileName) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        try {
            StringBuilder sb = new StringBuilder();
            String line = br.readLine();

            while (line != null) {
                sb.append(line);
                sb.append("\n");
                line = br.readLine();
            }
            return sb.toString();
        } finally {
            br.close();
        }
    }
    /**
     *   
     * @param wordList: List of words to be analyzed for unigram,bigram,trigram
     * @param percent: How many (most frequent) Xgrams are required for PERCENT coverage of the corpus
     * @param ngram : 1/2/3 for unigram/bigram/trigram
     */
    static void Analyze(List<String> wordList, int percent, int  ngram){
    	
    	if(ngram!=1 && ngram!=2 && ngram!=3){
    		System.out.println("NGram value is not correct plese input among 1,2 or 3");
    		return;
    	}
    	if(percent > 100 || percent < 0 ){
    		System.out.println("Percent Value is not correct,Please enter between 1 and 100");
    		return;
    	}
    	//get ngram from library function
    	Collection<String> nGramList= edu.stanford.nlp.util.StringUtils.getNgrams(wordList,ngram,ngram);
    	
    	//create map from ngram list
    	Map<String, Integer> nGramMap = new HashMap<String, Integer>();
    	
    	for(String it : nGramList){
			if(nGramMap.containsKey(it)) {
				nGramMap.put(it, nGramMap.get(it) + 1);				
			}else {
				nGramMap.put(it, 1);
			}
		}
    	
    	//get sorted nGrams
    	Map<String, Integer> sortednGramMap = sortByComparator(nGramMap, false);
    	int count = 0;
		System.out.println("---------------------------------------------");
		System.out.print("\nTop 10 frequent ");
		
		switch(ngram){
		case 1:
			System.out.print("Unigrams");
			break;
		case 2:
			System.out.print("Bigrams");
			break;
		case 3:
			System.out.print("Trigrams");
			break;			
		}
		
		System.out.print(" are: \n\n");
		
		for(Map.Entry<String, Integer> entry : sortednGramMap.entrySet()){
			String word = entry.getKey();
			Integer word_count = entry.getValue();
			System.out.println(word + " -> " + word_count);
			count += 1;
			if(count > 10){
				break;
			}
		}
		
		int totalnGram = nGramList.size();
		int reqnGram,nGramCount,nGramWord;
		
		nGramCount = nGramWord = 0;
		
		double dper = 1.0*percent;
		dper = dper/100;
		
		reqnGram = (int) Math.round(dper*totalnGram);
		
		for(Map.Entry<String, Integer> entry : sortednGramMap.entrySet()){			
			nGramCount += entry.getValue();
			nGramWord++;
			if(nGramCount >= reqnGram){
				break;
			}
		}
		
		System.out.print("\nTotal ");
		switch(ngram){
		case 1:
			System.out.print("Unigrams");
			break;
		case 2:
			System.out.print("Bigrams");
			break;
		case 3:
			System.out.print("Trigrams");
			break;
		}
		
		System.out.print(" Required for " + percent + "% coverage of corpus is " + nGramWord + " out of " + nGramMap.size() + "\n");		
    }
    	
    /**
     * 
     * @param text:input string to be lemmatized
     * @return : List<String> after performing lemmatization of input text
     */
	public static List<String> my_lemmatizer(String text){
				
		StanfordLemmatizer slem = new StanfordLemmatizer();
		List<String> lemmatizedText = slem.lemmatize(text);		
		return lemmatizedText;
	}
	        
	public static void main(String[] args) {
		
		String paragraph;
		
		try {
			paragraph = readFile("input.txt");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return;
		}
		
		Reader reader = new StringReader(paragraph);
		DocumentPreprocessor dp = new DocumentPreprocessor(reader);
		List<String> sentenceList = new ArrayList<String>();
		List<String> wordList = new ArrayList<String>();

		for(List<HasWord> sentence: dp ){			
			String sentenceString = edu.stanford.nlp.ling.Sentence.listToString(sentence);
			sentenceList.add(sentenceString.toString());			
			for (HasWord token : sentence) {
				wordList.add(token.word());							   
			}			
		}
					
		System.out.println("Number of Sentences are " + sentenceList.size());		
		
		System.out.println("Before Lemmatization Analysis is as below:");
		
		System.out.println("Total Number of Words are " + wordList.size());
		Analyze(wordList,90,1);
		Analyze(wordList,80,2);
		Analyze(wordList,70,3);
		
		System.out.println("----------------------------------------------------");
		System.out.println("----------------------------------------------------");
		
		System.out.println("Performing Lemmatization");		
		List<String> lemmatizedText = my_lemmatizer(paragraph);					
		System.out.println("Lemmatization Done");		
				
		
		System.out.println("After Lemmatization Analysis is as below:");
		System.out.println("Total Number of Words are " + lemmatizedText.size());
		Analyze(lemmatizedText,90,1);
		Analyze(lemmatizedText,80,2);
		Analyze(lemmatizedText,70,3);				
		
	}

}
