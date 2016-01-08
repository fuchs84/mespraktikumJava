package WekaAlgorithm;

import weka.core.Instances;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 08.01.16.
 */
public class DataGenerator {
    private final int offset = 4;
    String[][] patterns;

    public Instances[] buildTrain(String patternPath, String labelPath) throws Exception{
        String[][] patterns = readCSV(patternPath);
        String[][] labels = readCSV(labelPath);
        int numberOfInstances = patterns.length;
        int numberOfFeatures = patterns[0].length - offset;
        int numberOfLabels = labels[0].length - offset;
        Instances[] instancesArray = new Instances[numberOfLabels];
        String arff = "";
        for(int i = 0; i < numberOfLabels; i++) {
            arff = "";
            arff = arff + "@relation label" + i + "\n";
            for(int j = 0; j < numberOfFeatures; j++) {
                arff = arff + "@attribute " + j + " numeric\n";
            }
            arff = arff + "@attribute class {1,2,3}\n";
            arff = arff + "@data\n";
            for(int j = 0; j < numberOfInstances; j++) {
                for(int k = 0; k < numberOfFeatures; k++) {
                    arff = arff + patterns[j][k + offset] + ",";
                }
                arff = arff + labels[j][i + offset] + "\n";
            }
            InputStream stream = new ByteArrayInputStream(arff.getBytes("UTF-8"));
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            instancesArray[i] = new Instances(reader);
            instancesArray[i].setClassIndex(instancesArray[i].numAttributes() -1);
            reader.close();
        }
        return instancesArray;
    }

    public Instances buildClassify(String path) throws Exception {
        patterns = readCSV(path);
        int numberOfInstances = patterns.length;
        int numberOfFeatures = patterns[0].length - offset;
        String arff = "@relation classify\n";
        for(int j = 0; j < numberOfFeatures; j++) {
            arff = arff + "@attribute " + j + " numeric\n";
        }
        arff = arff + "@attribute class {1,2,3}\n";
        arff = arff + "@data\n";
        for(int j = 0; j < numberOfInstances; j++) {
            for(int k = 0; k < numberOfFeatures; k++) {
                arff = arff + patterns[j][k + offset] + ",";
            }
            arff = arff +  "?\n";
        }
        System.out.println(arff);
        InputStream stream = new ByteArrayInputStream(arff.getBytes("UTF-8"));
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));

        Instances instances = new Instances(reader);
        instances.setClassIndex(instances.numAttributes()-1);
        return instances;
    }

    public void saveResults(double[][] labels) {

    }


    private String[][] readCSV(String path) {
        ArrayList<String> data = new ArrayList<>();
        int lineIndex = 0;
        try {
            BufferedReader brP = new BufferedReader(new FileReader(path));
            String line;
            while ((line = brP.readLine()) != null) {
                data.add(line);
                lineIndex++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        String[][] separatedData = new String[lineIndex][];
        for(int i = 0; i < lineIndex; i++) {
            separatedData[i] = data.get(i).split(",");
        }
        return separatedData;
    }
}
