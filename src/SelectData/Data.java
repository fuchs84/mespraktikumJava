package SelectData;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public final class Data {
    private final double[][] label;
    private final double[][] pattern;
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

    public double[] computeDistribution(double[] labels) {
        int numberOfInstances = labels.length;
        double[] distribution = new double[3];
        for(int i = 0; i < labels.length; i++) {
            if((int)labels[i] == 1) {
                distribution[0]++;
            }
            if((int)labels[i] == 2) {
                distribution[1]++;
            }
            if((int)labels[i] == 3) {
                distribution[2]++;
            }
        }
        for(int i = 0; i < distribution.length; i++) {
            distribution[i] = distribution[i] / (double) numberOfInstances;
        }

        return distribution;
    }
}
