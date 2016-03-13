package SelectData;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Data-class
 */
public final class Data {
    /**
     * Arrays with complete-data set and split data-set (train and test)
     */
    private double[][] label;
    private double[][] pattern;
    public double [][] testPattern;
    public double [][] testLabel;
    public double [][] trainLabel;
    public double [][] trainPattern;

    /**
     * Percent/size of the train-set
     */
    private double split = 0.7;

    /**
     * Constructor for the data
     * @param label
     * @param pattern
     */
    public Data (double[][] label, double[][] pattern) {
        this.label = transpose(label);
        this.pattern = pattern;
        splitData(pattern, label);
    }

    /**
     * Method creates a train- and test-set with a percentage Split
     * @param pattern feature-set
     * @param label label-set
     */
    private void splitData(double[][] pattern, double[][] label) {
        int boarder = (int) (pattern.length*split);
        trainPattern = new double[boarder][];
        trainLabel = new double[boarder][];
        testPattern = new double[pattern.length - boarder][];
        testLabel = new double[pattern.length - boarder][];
        for (int i = 0; i < pattern.length; i++) {
            if(i < boarder) {
                trainPattern[i] = pattern[i];
                trainLabel[i] = label[i];
            } else {
                testPattern[i-boarder] = pattern[i];
                testLabel[i-boarder] = label[i];
            }
        }
        trainLabel = transpose(trainLabel);
        testLabel = transpose(testLabel);
    }

    /**
     * Method transpose a given matrix
     * @param data 2d-matrix
     * @return Transposed 2d-matrix
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
     * Getter-method for label-set
     * @return Label-set
     */
    public double[][] getLabel() {
        return label;
    }

    /**
     * Getter-method for feature-set
     * @return Feature-set
     */
    public double[][] getPattern() {
        return pattern;
    }

    /**
     * Setter-method for label-set
     * @param label Label-set
     */
    public void setLabel(double[][] label) {
        this.label = label;
    }

    /**
     * Setter-method for feature-set
     * @param pattern Feature-set
     */
    public void setPattern(double[][] pattern) {
        this.pattern = pattern;
    }
}
