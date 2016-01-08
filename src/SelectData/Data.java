package SelectData;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public final class Data {
    private double[][] label;
    private double[][] pattern;
    public double [][] testPattern;
    public double [][] testLabel;
    public double [][] trainLabel;
    public double [][] trainPattern;
    private double split = 0.7;

    public Data (double[][] label, double[][] pattern) {
        this.label = transpose(label);
        this.pattern = pattern;
        splitData(pattern, label);
    }

    public void shuffleData() {
        this.label = transpose(label);
        Random rnd = ThreadLocalRandom.current();
        for (int i = pattern.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            double[] a = pattern[index];
            double[] b = label[index];
            pattern[index] = pattern[i];
            label[index] = label[index];
            pattern[i] = a;
            label[i] = b;
        }
        this.label = transpose(label);
    }

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

    private double[][] transpose(double[][] data) {
        double[][] transpose = new double[data[0].length][data.length];
        for (int i = 0; i < transpose.length; i++) {
            for (int j = 0; j < transpose[0].length; j++) {
                transpose[i][j] = data[j][i];
            }
        }
        return transpose;
    }

    public double[][] getLabel() {
        return label;
    }

    public double[][] getPattern() {
        return pattern;
    }

    public void setLabel(double[][] label) {
        this.label = label;
    }

    public void setPattern(double[][] pattern) {
        this.pattern = pattern;
    }
}
