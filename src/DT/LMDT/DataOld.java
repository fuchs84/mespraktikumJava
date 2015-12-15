package DT.LMDT;

/**
 * Created by MatthiasFuchs on 14.12.15.
 */
public final class DataOld {
    private final double[] label;
    private final double[][] pattern;
    public double [][] testPattern;
    public double [] testLabel;
    public double [] trainLabel;
    public double [][] trainPattern;
    private double split = 0.7;

    public DataOld (double[] label, double[][] pattern) {
        this.label = label;
        this.pattern = pattern;
        splitData(pattern, label);
    }

    private void splitData(double[][] pattern, double[] label) {
        int boarder = (int) (pattern.length*split);
        trainPattern = new double[boarder][];
        trainLabel = new double[boarder];
        testPattern = new double[pattern.length - boarder][];
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
