package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;

/**
 * Created by MatthiasFuchs on 08.01.16.
 */

public class ClassifierJ48 extends Thread implements ClassifierInterface {
    private J48 tree;
    private boolean mode = false;
    private Instances instances;

    public ClassifierJ48() throws Exception {
        String[] options = new String[1];
        options[0] = "-U";
        tree = new J48();
        tree.setOptions(options);
    }

    public void setMode(boolean mode) {
        this.mode = mode;
    }

    public void setInstances(Instances instances) {
        this.instances = instances;
    }

    public void train() throws Exception{
        tree.buildClassifier(instances);
    }

    public double[] classify() throws Exception {
        double[] classify = new double[instances.numInstances()];
        for(int i = 0; i < instances.numInstances(); i++) {
            classify[i] = tree.classifyInstance(instances.instance(i));
        }
        return classify;
    }

    public Classifier getClassifier() {
        return tree;
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
