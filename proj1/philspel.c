/*
 * Include the provided hashtable library
 */
#include "hashtable.h"

/*
 * Include the header file
 */
#include "philspel.h"

/*
 * Standard IO and file routines
 */
#include <stdio.h>

/*
 * General utility routines (including malloc())
 */
#include <stdlib.h>

/*
 * Character utility routines.
 */
#include <ctype.h>

/*
 * String utility routine
 */
#include <string.h>

/*
 * this hashtable stores the dictionary
 */
HashTable *dictionary;

/*
 * the MAIN routine.  You can safely print debugging information
 * to standard error (stderr) and it will be ignored in the grading
 * process in the same way which this does.
 */
int main(int argc, char **argv){
  if(argc != 2){
    fprintf(stderr, "Specify a dictionary\n");
    return 0;
  }
  /*
   * Allocate a hash table to store the dictionary
   */
  fprintf(stderr, "Creating hashtable\n");
  dictionary = createHashTable(2255, &stringHash, &stringEquals);

  fprintf(stderr, "Loading dictionary %s\n", argv[1]);
  readDictionary(argv[1]);
  fprintf(stderr, "Dictionary loaded\n");

  fprintf(stderr, "Processing stdin\n");
  processInput();

  /* main in C should always return 0 as a way of telling
     whatever program invoked this that everything went OK
     */
  return 0;
}


/*
 * You need to define this function. void *s can be safely casted
 * to a char * (null terminated string) which is done for you here for
 * convenience.
 *
 * This hashing function is cited from this url: http://www.cse.yorku.ca/~oz/hash.html.
 */

unsigned int stringHash(void *s){
    if (!s) {
        fprintf(stderr,"Need to define stringHash\n");
        exit(0);
    }

    unsigned long hash = 5381;
    int c;
    char *character = (char *) s;
    c = *character;
    while (c == *character) {
        character += 1;
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }

    return hash;
}



/*
 * You need to define this function.  It should return a nonzero
 * value if the two strings are identical (case sensitive comparison)
 * and 0 otherwise.
 */
int stringEquals(void *s1, void *s2){
    if (!s1 || !s2) {
        fprintf(stderr,"Need to define stringEquals\n");
        exit(0);
    }

    char *string1 = (char *) s1;
    char *string2 = (char *) s2;

    int compResult = strcmp(string1, string2);
    if (compResult == 0) {
        return 1;
    } else {
        return 0;
    }
}

/*
 * this function should read in every word in the dictionary and
 * store it in the dictionary.  You should first open the file specified,
 * then read the words one at a time and insert them into the dictionary.
 * Once the file is read in completely, exit.  You will need to allocate
 * (using malloc()) space for each word.  As described in the specs, you
 * can initially assume that no word is longer than 70 characters.  However,
 * for the final 20% of your grade, you cannot assumed that words have a bounded
 * length.  You can NOT assume that the specified file exists.  If the file does
 * NOT exist, you should print some message to standard error and call exit(0)
 * to cleanly exit the program. Since the format is one word at a time, with
 * returns in between, you can
 * safely use fscanf() to read in the strings.
 */
void readDictionary(char *filename){
    FILE *file;
    file = fopen(filename, "r");

    if (!file) {
        fprintf(stderr,"Need to define readDictionary\n");
        exit(0);
    }

    int charSize = 71;
    int index = 0;
    int singleChar;
    singleChar = fgetc(file);
    char *word = malloc(charSize * sizeof(char));
    while (singleChar != -1 || singleChar != EOF) {
        if (index >= charSize) {
            charSize *= 2;
            char* tmp;
            tmp = realloc(word, charSize * sizeof(char));
            if (!tmp) {
                fprintf(stderr, "Realloc fasiled");
                exit(0);
            } else {
                *word = *tmp;
            }
            free(tmp);
        }
        if ((singleChar >= 'A' && singleChar <= 'Z') || (singleChar >= 'a' && singleChar <= 'z')) {
            word[index] = (char) singleChar;
            index += 1;
        } else {
            word[index] = '\0';
            insertData(dictionary, word, word);
            word = (char*) malloc(charSize * sizeof(char));
            index = 0;
        }
        singleChar = fgetc(file);
    }

    word[index] = '\0';
    insertData(dictionary, word, word);

    fclose(file);
    free(word);
}

/*
 * This should process standard input and copy it to standard output
 * as specified in specs.  EG, if a standard dictionary was used
 * and the string "this is a taest of  this-proGram" was given to
 * standard input, the output to standard output (stdout) should be
 * "this is a teast [sic] of  this-proGram".  All words should be checked
 * against the dictionary as they are input, again with all but the first
 * letter converted to lowercase, and finally with all letters converted
 * to lowercase.  Only if all 3 cases are not in the dictionary should it
 * be reported as not being found, by appending " [sic]" after the
 * error.
 *
 * Since we care about preserving whitespace, and pass on all non alphabet
 * characters untouched, and with all non alphabet characters acting as
 * word breaks, scanf() is probably insufficent (since it only considers
 * whitespace as breaking strings), so you will probably have
 * to get characters from standard input one at a time.
 *
 * As stated in the specs, you can initially assume that no word is longer than
 * 70 characters, but you may have strings of non-alphabetic characters (eg,
 * numbers, punctuation) which are longer than 70 characters. For the final 20%
 * of your grade, you can no longer assume words have a bounded length.
 */

void processInput(){
    int i, j;
    int charSize = 70;
    int inpWord;
    int index = 0;
    char *finalword = (char*) malloc(charSize*sizeof(char));

    inpWord = getchar();
    char *transformed;
    while(inpWord != -1 || inpWord != EOF) {

        if(index >= charSize) {
            charSize *= 2;
            char *tmp;
            tmp = (char*) realloc(finalword, charSize * sizeof(char));
            if (tmp) {
                finalword = tmp;
            } else {
                fprintf(stderr, "Realloc failed");
                exit(0);
            }
            free(tmp);
        }

        if ((inpWord >= 'a'&& inpWord <= 'z') || (inpWord >= 'A' && inpWord <= 'Z')) {
            finalword[index] = (char) inpWord;
            index += 1;
        } else {
            finalword[index] = '\0';
            index = 0;

            if(finalword[0] != '\0' && !findData(dictionary, finalword)) {
                transformed = (char *) malloc(charSize * sizeof(char));
                transformed[0] = finalword[0];
                if (strlen(finalword) > 0 ) {
                    for(i = 1; i <= strlen(finalword); i += 1) {
                        transformed[i] = (char) tolower(finalword[i]);
                    }
                }

                if(finalword[0] != '\0' && !findData(dictionary, transformed)) {
                    transformed[0] = (char) tolower(transformed[0]);
                    if(!findData(dictionary, transformed)) {
                        printf("%s ", finalword);
                        printf("%s", "[sic]");
                        printf("%c", (char) inpWord);
                    } else {
                        printf("%s%c", finalword, (char) inpWord);
                    }
                } else {
                    printf("%s%c", finalword, (char) inpWord);
                }
                free(transformed);
            } else {
                printf("%s%c", finalword, (char) inpWord);
            }
        }

        inpWord = getchar();
    }

    finalword[index] = '\0';
    if(finalword[0] != '\0' && !findData(dictionary, finalword)) {
        transformed = (char *) malloc(charSize*sizeof(char));
        transformed[0] = finalword[0];
        if (strlen(finalword) > 0 ) {
            for(j = 1; j <= strlen(finalword); j += 1) {
                transformed[i] = (char) tolower(finalword[i]);
            }
        }
        if(finalword[0] != '\0' && !findData(dictionary, transformed)) {
            transformed[0] = (char) tolower(transformed[0]);
            if(!findData(dictionary, transformed)) {
                printf("%s ", finalword);
                printf("%s", "[sic]");
            } else {
                printf("%s", finalword);
            }
        } else {
            printf("%s", finalword);
        }
        free(transformed);
    } else {
        printf("%s", finalword);
    }
    free(finalword);
}

