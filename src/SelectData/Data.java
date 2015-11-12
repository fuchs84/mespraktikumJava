package SelectData;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public final class Data {
    private double max;
    private final double[] label;
    private final double[][] pattern;
    public double [][] testPattern;
    public double [] testLabel;
    public double [] trainLabel;
    public double [][] trainPattern;
    private double split = 0.6;

    public Data (double[] label, double[][] pattern, double max) {
        this.label = label;
        this.pattern = pattern;
        this.max = max;
        splitData();
    }

    private void splitData() {
        int boarder = (int) (pattern.length*split);
        trainPattern = new double[boarder][pattern[0].length];
        trainLabel = new double[boarder];
        testPattern = new double[pattern.length - boarder][pattern[0].length];
        testLabel = new double[pattern.length - boarder];
        for (int i = 0; i < pattern.length; i++) {
            if(i < boarder) {
                trainPattern[i] = pattern[i];
                trainLabel[i] = label[i];
            } else {
                testPattern[i-boarder] = pattern[i];
                testLabel[i-boarder] = label[i];
            }
        }
    }

    public double[] getLabel() {
        return label;
    }

    public double[][] getLabelForMLP (double[] label) {
        int length = 8, value;
        double [][] labelMLP = new double[label.length][length];

        for (int i = 0; i < label.length; i ++) {
            value = (int) label[i];
            labelMLP[i][value-1] = 1.0;
        }
        return labelMLP;
    }

    public double[][] getPattern() {
        return pattern;
    }

    public double getMax() {
        return max;
    }

    public double [][] getScaledPattern(double [][] pattern) {
        double[][] scaledPattern = new double[pattern.length][pattern[0].length];
        for (int i = 0; i < pattern.length; i++) {
            for (int j = 0; j < pattern[0].length; j++) {
                scaledPattern[i][j] = pattern[i][j]/max;
            }
        }
        return scaledPattern;
    }

}
