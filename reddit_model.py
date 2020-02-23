from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, count, avg, from_unixtime, desc
from pyspark.sql.types import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from cleantext import sanitize
import os

# Load data from parquet if available, else load data while storing in parquet
def load_parquet(context, parquet_dir, path):
    df = None
    if os.path.exists(parquet_dir):
        df = context.read.parquet(parquet_dir)
    else:
        dirpath, extension = os.path.splitext(path)
        if extension == ".csv":
            df = context.read.csv(path, header=True)
        elif extension == ".bz2":
            df = context.read.json(path)
        else:
            print("Unsupported file type")
        df.write.parquet(parquet_dir)

    return df

def get_combined_ngrams(comment_body):
    results = sanitize(comment_body)
    unigrams = results[1].split(' ')
    bigrams = results[2].split(' ')
    trigrams = results[3].split(' ')
    return (unigrams + bigrams + trigrams)

# Returns dataframe with columns {comment_id, split_ngrams, labeldjt}
# (comment id, list of all split ngrams, labeled rating for donald trump)
def load_ngrams(context, comments_df, labeled_data_df):
    # Initialize UDF
    ngrams_udf = udf(lambda body: get_combined_ngrams(body), ArrayType(StringType(), False))
    # Join labeled data and comments on comment_id.
    ngrams = comments_df.join(labeled_data_df, comments_df.id == labeled_data_df.Input_id) \
                .select((labeled_data_df.Input_id).alias('comment_id'), \
                        (ngrams_udf(comments_df.body)).alias('split_ngrams'), \
                        labeled_data_df.labeldjt)

    return ngrams

def sparsify(ngrams_df, model):
    if model is None:
        # TASK 6a: Binary CountVectorizer
        cv = CountVectorizer(minDF=10, binary=True,
                            inputCol="split_ngrams", outputCol="sparse_vector")
        model = cv.fit(ngrams_df)
    
    sparsified = model.transform(ngrams_df)
    return model, sparsified

def compute_sentiments(sparsified):
    #TASK 6b: Add 2 new columns for positive or negative sentiments
    positive_udf = udf(lambda djt: 1 if djt == "1"  else 0, IntegerType())
    negative_udf = udf(lambda djt: 1 if djt == "-1" else 0, IntegerType())
    #print(sparse_vector.columns)
    final_result = sparsified.select('*',
                                      	(positive_udf(sparsified.labeldjt)).alias("positive"),
                                       	(negative_udf(sparsified.labeldjt)).alias("negative"))
    return final_result
    # final_result.show()


def train_model(pos, neg): 
    # Initialize two logistic regression models.
    # Replace labelCol with the column containing the label, and featuresCol with the column containing the features.
    poslr = LogisticRegression(labelCol="positive", featuresCol="sparse_vector", maxIter=10)
    neglr = LogisticRegression(labelCol="negative", featuresCol="sparse_vector", maxIter=10)
    # This is a binary classifier so we need an evaluator that knows how to deal with binary classifiers.
    posEvaluator = BinaryClassificationEvaluator(labelCol="positive")
    negEvaluator = BinaryClassificationEvaluator(labelCol="negative")
    # There are a few parameters associated with logistic regression. We do not know what they are a priori.
    # We do a grid search to find the best parameters. We can replace [1.0] with a list of values to try.
    # We will assume the parameter is 1.0. Grid search takes forever.
    posParamGrid = ParamGridBuilder().addGrid(poslr.regParam, [1.0]).build()
    negParamGrid = ParamGridBuilder().addGrid(neglr.regParam, [1.0]).build()
    # We initialize a 5 fold cross-validation pipeline.
    posCrossval = CrossValidator(
        estimator=poslr,
        evaluator=posEvaluator,
        estimatorParamMaps=posParamGrid,	
        numFolds=5) # CHANGE TO 2 FOR DEBUG. DELETE /project2 BEFORE RETRAINING.
    negCrossval = CrossValidator(
        estimator=neglr,
        evaluator=negEvaluator,
        estimatorParamMaps=negParamGrid,
        numFolds=5) # CHANGE TO 2 FOR DEBUG. DELETE /project2 BEFORE RETRAINING.

    # Although crossvalidation creates its own train/test sets for
    # tuning, we still need a labeled test set, because it is not
    # accessible from the crossvalidator (argh!)
    # Split the data 50/50
    posTrain, posTest = pos.randomSplit([0.5, 0.5])
    negTrain, negTest = neg.randomSplit([0.5, 0.5])
    # Train the models
    print("Training positive classifier...")
    posModel = posCrossval.fit(posTrain)
    print("Training negative classifier...")
    negModel = negCrossval.fit(negTrain)

    # Once we train the models, we don't want to do it again. We can save the models and load them again later.
    posModel.save("project2/pos.model")
    negModel.save("project2/neg.model")




def get_final_data(comments, submissions):
    # Define UDF for stripping 3-char prefix from link_id
    strip_link = udf(lambda link: link[3:], StringType())
    ngrams_udf = udf(lambda body: get_combined_ngrams(body), ArrayType(StringType(), False))
    
    # Reread comments. get:
    #   1. comment ID
    #   2. comment text
    #   3. timestamp
    #   4. post title (join comments.link_id (with t3_ stripped, using a UDF) on submissions.id)
    #   5. state (flair)
    # Remove all comments that contain the word "/s" or that start with &gt.
    final_columns = comments.filter(~comments.body.rlike("^(\s*)(&gt)(\s.*)?$|^(.*\s)?(/s)(\s.*)?$"))\
                            .join(submissions, strip_link(comments.link_id) == submissions.id)\
                            .select(comments.id, 
                                    comments.created_utc, 
                                    comments.author_flair_text,                                    
                                    comments.score.alias('comment_score'),
                                    (ngrams_udf(comments.body)).alias('split_ngrams'),
                                    submissions.id.alias('submission_id'),
                                    submissions.title,
                                    submissions.score.alias('submission_score'))
                            
    return final_columns

# regex: matches string that start with &gt or contains /s 
# (but not _&gt or &gt_, or _/s or /s_ for any non-whitespace _)
def main(context):
    """Main function takes a Spark SQL context."""

    # TASK 1: Load parquet if available, if not, load file then write parquet
    comments = load_parquet(context, "./comments-minimal.parquet",
                                        "comments-minimal.json.bz2")
    submissions = load_parquet(context, "./submissions.parquet",
                                        "submissions.json.bz2")
    labeled_data = load_parquet(context, "./labeled_data.parquet", "labeled_data.csv")

    # TASKS 2, 4, 5: Generate split ngrams joined on comment id and trump rating
    ngrams = load_ngrams(context, comments, labeled_data)

    # TASK 6A: Convert to sparse vector
    cv_model, sparsified = sparsify(ngrams, None)

    # TASK 6B: Add positive/negative label columns
    training_set = compute_sentiments(sparsified)
    
    # TASK 7: Train model with k-fold cross validation, 20% test 80% train, k = 5.
    # Do not retrain if already exists. 
    # To reset training data, delete directory /project2.
    training_set_pos = training_set.select(training_set.sparse_vector,
                                           training_set.positive)
    training_set_neg = training_set.select(training_set.sparse_vector,
                                           training_set.negative)
    if not os.path.exists("project2"):
        train_model(training_set_pos, training_set_neg)

	# TASKS 8, 9: Reread comments (and save to parquet if not already).
    #   Removes all comments with /s or that start with &gt.
    if os.path.exists("./final-data.parquet"):
        print("Reading final data from parquet...")
        final_data = context.read.parquet("./final-data.parquet")
    else:
        final_data = get_final_data(comments, submissions)
        print("Writing final data...")
        final_data.write.parquet("./final-data.parquet")
    

    dontcare, final_data_processed = sparsify(final_data, cv_model)
    # final_data_processed.show()

    # TASK 9: Apply classifier to transformed data ("final_data_processed").
    #   Ceiling the probabilities for positive/negative sentiment.
    posModel = CrossValidatorModel.load("project2/pos.model")
    negModel = CrossValidatorModel.load("project2/neg.model")     
    pos_ceiling = udf(lambda prob: 1 if float(prob[1]) > 0.2  else 0, IntegerType())
    neg_ceiling = udf(lambda prob: 1 if float(prob[1]) > 0.25 else 0, IntegerType())

    # USE THIS FOR SAMPLING
    final_sampled = final_data_processed.sample(False, 0.02, None)
    pos_results = posModel.transform(final_sampled)

    # USE THIS FOR FULL SET
    # pos_results = posModel.transform(final_data_processed)

    pos_selected = pos_results.select(pos_results.id,
                                        pos_results.created_utc,
                                        pos_results.author_flair_text,
                                        pos_results.comment_score,
                                        pos_results.sparse_vector,
                                        pos_results.submission_id,
                                        pos_results.title,
                                        pos_results.submission_score,
                                        pos_ceiling(pos_results.probability).alias("pos"))
    
    neg_results = negModel.transform(pos_selected)
    final_results = neg_results.select(neg_results.id,
                                        neg_results.created_utc,
                                        neg_results.author_flair_text,
                                        neg_results.comment_score,
                                        neg_results.sparse_vector,
                                        neg_results.submission_id,
                                        neg_results.title,
                                        neg_results.submission_score,
                                        neg_results.pos,
                                        neg_ceiling(neg_results.probability).alias("neg"))

    # print("Total comments: ", comments.count())
    # print("Total final rows: ", final_results.count()) # slightly less than total comments bc of removed comments
    # final_results.show()

    # TASK 10: computations. Delete .csv's to recompute.
    #	10.1: Percentage of comments across all submissions
    submission_percents = final_results.select("submission_id", "title", "pos", "neg")\
                                       .groupBy("submission_id", "title")\
                                       .agg(avg("pos").alias("pos_pct"), avg("neg").alias("neg_pct"))\
                                       .filter(count(final_results.submission_id) > 10)

    submission_percents.orderBy(desc("pos_pct")).limit(10).write.csv("top_pos_subs.csv", header=True)
    submission_percents.orderBy(desc("neg_pct")).limit(10).write.csv("top_neg_subs.csv", header=True)

    # 	10.2: Percentage of comments across all days
    if not os.path.exists("time_data.csv"):
        day_percents = final_results.select("pos", "neg", from_unixtime("created_utc", "yyyy-MM-dd").alias("date"))\
                                    .groupBy("date")\
                                    .agg(avg("pos").alias("Positive"), avg("neg").alias("Negative"))
        day_percents.coalesce(1).write.csv("time_data.csv", header=True)

    #	10.3: Percentage of comments across states
    if not os.path.exists("state_data.csv"):
        states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
                'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas',
                'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
                'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina',
                'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
                'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
        state_percents = final_results.where(final_results.author_flair_text.isin(states))\
                                    .select("pos", "neg", final_results.author_flair_text.alias("state"))\
                                    .groupBy("state")\
                                    .agg(avg("pos").alias("Positive"), avg("neg").alias("Negative"))
        state_percents.coalesce(1).write.csv("state_data.csv", header=True)
    
	# 	10.4: Percentage of comments across comment score and across submission score
    if not os.path.exists("submission_score.csv"):
        submission_percents = final_results.select("submission_score", "pos", "neg")\
                                        .groupBy("submission_score")\
                                        .agg(avg("pos").alias("Positive"), avg("neg").alias("Negative"))
        submission_percents.coalesce(1).write.csv("submission_score.csv", header=True)
    
    if not os.path.exists("comment_score.csv"):
        comment_percents = final_results.select("comment_score", "pos", "neg")\
                                        .groupBy("comment_score")\
                                        .agg(avg("pos").alias("Positive"), avg("neg").alias("Negative"))
        comment_percents.coalesce(1).write.csv("comment_score.csv", header=True)


if __name__ == "__main__":
    conf = SparkConf().setAppName("CS143 Project 2B")
    conf = conf.setMaster("local[*]")
    sc   = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.addPyFile("cleantext.py")
    main(sqlContext)