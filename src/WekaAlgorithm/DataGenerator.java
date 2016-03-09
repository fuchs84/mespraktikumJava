package WekaAlgorithm;

import weka.core.Instances;

import java.io.*;
import java.util.ArrayList;

/**
 * Data generator for the classifiers
 */
public class DataGenerator {
    private final int offset = 4;
    String[][] patterns;

    /**
     * Method builds the train-instances for each classifier/mistake
     * @param patternPath Pattern storage path
     * @param labelPath Label storage path
     * @return Array with the instances for each classifier/mistake
     * @throws Exception to invoking method
     */
    public Instances[] buildTrain(String patternPath, String labelPath) throws Exception{
        String[][] patterns = readCSV(patternPath);
        String[][] labels = readCSV(labelPath);
        int numberOfPatternInstances = patterns.length;
        int numberOfFeatures = patterns[0].length - offset;
        int numberOfLabels = labels[0].length - offset;
        Instances[] instancesArray = new Instances[numberOfLabels];
        for(int i = 0; i < numberOfLabels; i++) {
            System.out.println("Instances start" + i);
            StringBuilder arff = new StringBuilder();
            arff.append("@relation label" + i + "\n");
            for(int j = 0; j < numberOfFeatures; j++) {
                arff.append("@attribute " + j + " numeric\n");
            }
            arff.append("@attribute class {1,2,3}\n");
            arff.append("@data\n");
            for(int j = 0; j < numberOfPatternInstances; j++) {
                for(int k = 0; k < numberOfFeatures; k++) {
                    arff.append(patterns[j][k + offset] + ",");
                }
                arff.append(labels[j][i+offset]+ "\n");
            }
            System.out.println("Instances finish" + i);
            InputStream stream = new ByteArrayInputStream(arff.toString().getBytes("UTF-8"));
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            instancesArray[i] = new Instances(reader);
            instancesArray[i].setClassIndex(instancesArray[i].numAttributes() - 1);
            reader.close();
        }
        return instancesArray;
    }

    /**
     * Method builds classifying instance (one for all classifiers)
     * @param path Pattern storage path
     * @return Instance for the classifiers
     * @throws Exception to invoking method
     */
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

    /**
     * Method saves the classified results
     * @param labels Classified Labels of all mistakes
     * @throws Exception to invoking Method
     */
    public void saveResults(double[][] labels) throws Exception {
        labels = transpose(labels);
        FileWriter fw = new FileWriter("results.csv");
        for(int i = 0; i < labels.length; i++) {
            for(int j = 0; j < labels[0].length +offset; j++) {
                if(j < offset) {
                    fw.append(patterns[i][j]);
                    fw.append(",");
                } else if (j == labels[0].length + offset -1) {
                    fw.append(Double.toString(labels[i][j-offset]));
                }
                else {
                    fw.append(Double.toString(labels[i][j-offset]));
                    fw.append(",");
                }
            }
            fw.append("\n");
        }
        fw.close();
    }

    /**
     * Method transposes the 2d-matrix data
     * @param data 2d-matrix
     * @return transposed 2d-matrix
     */
    private double[][] transpose(double[][] data) {
        double[][] transpose = new double[data[0].length][data.length];
        for (int i = 0; i < transpose.length; i++) {
            for (int j = 0; j < transpose[0].length; j++) {
                transpose[i][j] = data[j][i];
            }
        }
        return transpose;
    }

    /**
     * Methods reads the CSV-file
     * @param path storage path of the CSV-file
     * @return 2d-array with the read data
     */
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
