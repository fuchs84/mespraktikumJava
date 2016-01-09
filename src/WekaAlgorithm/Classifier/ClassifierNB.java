package WekaAlgorithm.Classifier;


import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 * Created by MatthiasFuchs on 09.01.16.
 */
public class ClassifierNB extends AbstractClassifier implements Runnable {
    private NaiveBayes classifier;

    public ClassifierNB() throws Exception {
        classifier = new NaiveBayes();
    }

    public void train() throws Exception{
        classifier.buildClassifier(instances);
    }

    public double[] classify() throws Exception {
        classified = new double[instances.numInstances()];
        for(int i = 0; i < instances.numInstances(); i++) {
            classified[i] = classifier.classifyInstance(instances.instance(i));
        }
        return classified;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    @Override
    public void run() {
        if(mode) {
            try {
                classify();
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            try {
                train();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}