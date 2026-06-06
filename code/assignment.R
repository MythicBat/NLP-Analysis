install.packages(c("tidyverse", "tm", "SnowballC", "proxy", "cluster", "syuzhet", "igraph", "ggraph", "tidygraph", "ggplot2"))

library(tidyverse)
library(tm)
library(SnowballC)
library(proxy)  
library(cluster)
library(syuzhet)
library(igraph)
library(ggraph)
library(tidygraph)
library(ggplot2)

# Load the dataset
corpus_df <- read.csv("data/corpus.csv", stringsAsFactors = FALSE)

# Check structure
str(corpus_df)
head(corpus_df)

table(corpus_df$genre)

# Check word counts
corpus_df$word_count <- sapply(strsplit(corpus_df$text, "\\s+"), length)

summary(corpus_df$word_count)

# QUESTION 3: Text Processing and Document-Term Matrix

# Create text corpus from the text column
text_corpus <- VCorpus(VectorSource(corpus_df$text))

# Basic text cleaning
text_corpus <- tm_map(text_corpus, content_transformer(tolower))
text_corpus <- tm_map(text_corpus, removePunctuation)
text_corpus <- tm_map(text_corpus, removeNumbers)
text_corpus <- tm_map(text_corpus, removeWords, stopwords("english"))
text_corpus <- tm_map(text_corpus, stripWhitespace)

# stemming
text_corpus <- tm_map(text_corpus, stemDocument)

# Create initial DTM
dtm <- DocumentTermMatrix(text_corpus)

dtm

# Trial-and-error sparse term removal
# Increase/decrease this value unitl around 25 terms remain
dtm_sparse <- removeSparseTerms(dtm, 0.20)

dtm_sparse

# Convert dtm to matrix
dtm_matrix <- as.matrix(dtm_sparse)

# Add document IDs as row names
rownames(dtm_matrix) <- corpus_df$doc_id

# View final tokens
colnames(dtm_matrix)

write.csv(dtm_matrix, "data/dtm_matrix.csv", row.names = TRUE)

# Save token frquency table
token_freq <- colSums(dtm_matrix)
token_freq_df <- data.frame(
    token = names(token_freq),
    frequency = as.numeric(token_freq)
)  %>%
    arrange(desc(frequency))

write.csv(token_freq_df, "data/token_freq.csv", row.names = FALSE)

# Print number of final tokens
cat("Number of final tokens:", ncol(dtm_matrix), "\n")
