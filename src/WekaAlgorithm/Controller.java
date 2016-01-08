package WekaAlgorithm;

import WekaAlgorithm.Classifier.ClassifierJ48;
import weka.core.Instances;

/**
 * Created by MatthiasFuchs on 08.01.16.
 */
public class Controller {
    private DataGenerator dataGenerator;
    private ClassifierJ48[] j48s;


    public Controller() {
        dataGenerator = new DataGenerator();
    }

    public void train(String patternPathAll, String labelPathAll, String patternPathPass, String labelPathPass) {
        try {
            Instances[] allInstances = dataGenerator.buildTrain(patternPathAll, labelPathAll);
            Instances[] passInstances = dataGenerator.buildTrain(patternPathPass, labelPathPass);
            j48s = new ClassifierJ48[allInstances.length + passInstances.length];
            for(int i = 0; i < j48s.length; i++) {
                j48s[i] = new ClassifierJ48();
                j48s[i].setMode(false);
                if(i < allInstances.length) {
                    j48s[i].setInstances(allInstances[i]);
                } else {
                    j48s[i].setInstances(passInstances[i-allInstances.length]);
                }
                j48s[i].start();
                System.out.println(i);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void classify(String patternPath) {
        double[][] labels = new double[j48s.length][];
        try {
            Instances instances = dataGenerator.buildClassify(patternPath);

            for(int i = 0; i < j48s.length; i++) {
                j48s[i].setInstances(instances);
                labels[i] = j48s[i].classify();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void evaluation(String type) {

    }
}
