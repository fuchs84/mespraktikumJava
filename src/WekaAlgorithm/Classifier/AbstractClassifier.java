package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Abstract class specifies all methods for the classifiers
 */
public abstract class AbstractClassifier implements Runnable {

    protected Instances instances;
    protected double[] classified;

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
