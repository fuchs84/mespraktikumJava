package SelectData;
import java.lang.reflect.Array;
import java.util.ArrayList;

/**
 * Created by Sebastian on 12.11.2015.
 */
public class Crossvalidation {

    public ArrayList<ArrayList> crossvalidate(double [][] inputpattern, double [] inputlabel, int splits){
        ArrayList<ArrayList> data = new ArrayList<>();
        ArrayList<double[][]> patternlist = new ArrayList();
        ArrayList<double[]> labelist = new ArrayList();

        int lengthsplit = (inputlabel.length/splits);
        for (int i = 0; i < splits; i++) {
            double pattern[][]= new double[lengthsplit][inputpattern[0].length];
            double label[]= new double[lengthsplit];
            int k = i*lengthsplit;
            for (int j = 0; j <lengthsplit ; j++) {
                for (int l = 0; l <inputpattern[0].length ; l++) {
                   pattern[j][l] = inputpattern[k+j][l];
                }
                label[j]=inputlabel[k+j];
            }
            patternlist.add(pattern);
            labelist.add(label);
        }
        data.add(patternlist);
        data.add(labelist);
        return data;
    }

    public ArrayList<ArrayList> randomDataSplit(double [][] inputpattern, double [] inputlabel, double percent) {
        ArrayList<ArrayList> data = new ArrayList<>();
        ArrayList<double[][]> patterns = new ArrayList<>();
        ArrayList<double[]> labels = new ArrayList<>();
        ArrayList<double[]> tempPatternsTrain = new ArrayList<>();
        ArrayList<double[]> tempPatternsTest = new ArrayList<>();
        ArrayList<Double> tempLabelsTrain = new ArrayList<>();
        ArrayList<Double> tempLabelsTest = new ArrayList<>();
        for(int i = 0; i < inputpattern.length; i++) {
            if(Math.random() < percent) {
                tempPatternsTrain.add(inputpattern[i]);
                tempLabelsTrain.add(inputlabel[i]);
            } else {
                tempPatternsTest.add(inputpattern[i]);
                tempLabelsTest.add(inputlabel[i]);
            }
        }
        double[][] trainPatterns = new double[tempPatternsTrain.size()][];
        double[][] testPatterns = new double[tempPatternsTest.size()][];
        double[] trainLabels = new double[tempLabelsTrain.size()];
        double[] testLabels = new double[tempLabelsTest.size()];

        for(int i = 0; i < tempLabelsTrain.size(); i++) {
            trainPatterns[i] = tempPatternsTrain.get(i);
            trainLabels[i] = tempLabelsTrain.get(i);
        }
        for(int i = 0; i < tempLabelsTest.size(); i++) {
            testPatterns[i] = tempPatternsTest.get(i);
            testLabels[i] = tempLabelsTest.get(i);
        }
        patterns.add(trainPatterns);
        patterns.add(testPatterns);
        labels.add(trainLabels);
        labels.add(testLabels);
        data.add(patterns);
        data.add(labels);
        return data;
    }
}





