package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Created by MatthiasFuchs on 09.01.16.
 */
public abstract class AbstractClassifier implements Runnable {
    protected boolean mode = false;
    protected Instances instances;
    protected double[] classified;

    public void setMode(boolean mode) {
        this.mode = mode;
    }

    public void setInstances(Instances instances) {
        this.instances = instances;
    }

    public double[] getClassified() {
        return classified;
    }

    public abstract void train() throws Exception;

    public abstract double[] classify() throws Exception;

    public abstract Classifier getClassifier();

    @Override
    public abstract void run();
}
