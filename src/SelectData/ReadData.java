package SelectData;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Data read from CSV-files
 */
public class ReadData {

    /**
     * Method reads two CSV-files (pattern-file and label-file)
     * @param patternPath Storage path feature-file
     * @param labelPath Storage path label-file
     * @return data-object (see Data-Class)
     */
    public Data readCSVs(String patternPath, String labelPath) {
        double[][] pattern;
        double[][] label;
        ArrayList<String> patternData = new ArrayList<>();
        ArrayList<String> labelData = new ArrayList<>();
        int lineIndex = 0;
        try {
            BufferedReader brP = new BufferedReader(new FileReader(patternPath));
            BufferedReader brL = new BufferedReader(new FileReader(labelPath));


            String patternLine;
            String labelLine;

            while ((patternLine = brP.readLine()) != null && (labelLine = brL.readLine()) != null) {
                patternData.add(patternLine);
                labelData.add(labelLine);
                lineIndex++;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        //Offset instances
        int offset = 0;

        //Offset timestamps
        int offsetLabels = 4;
        int offsetPatterns = 4;

        String[] labelParts = labelData.get(0).split(",");
        String[] patternParts = patternData.get(0).split(",");
        pattern = new double[lineIndex - offset][patternParts.length-offsetPatterns];
        label = new double[lineIndex - offset][labelParts.length-offsetLabels];

        for(int i = offset; i < lineIndex; i++) {
            patternParts = patternData.get(i).split(",");
            labelParts = labelData.get(i).split(",");
            if(pattern[0].length > label[0].length) {
                for(int j = 0; j < pattern[0].length; j++) {
                    pattern[i-offset][j] = Double.parseDouble(patternParts[j+offsetPatterns]);
                    if(j < label[0].length) {
                        label[i-offset][j] = Double.parseDouble(labelParts[j+offsetLabels]);
                    }
                }
            } else {
                for(int j = 0; j < label[0].length; j++) {
                    label[i-offset][j] = Double.parseDouble(labelParts[j+offsetLabels]);
                    if(j < pattern[0].length) {
                        pattern[i-offset][j] = Double.parseDouble(patternParts[j+offsetPatterns]);
                    }
                }
            }
        }
        return new Data(label, pattern);
    }
}
