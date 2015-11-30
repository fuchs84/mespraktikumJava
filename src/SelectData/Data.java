package SelectData;

/**
 * Created by MatthiasFuchs on 12.11.15.
 */
public final class Data {
    private final double[] label;
    private final double[][] pattern;
    public double [][] testPattern;
    public double [] testLabel;
    public double [] trainLabel;
    public double [][] trainPattern;
    private double split = 0.7;

    public Data (double[] label, double[][] pattern) {
        this.label = label;
        this.pattern = pattern;
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

    public double[][] getPattern() {
        return pattern;
    }
}
