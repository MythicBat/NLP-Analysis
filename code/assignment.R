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

# ============================================================
# QUESTION 4: Hierarchical Clustering of Documents
# ============================================================

install.packages("dendextend")
library(dendextend)

# Cosine distance
cos_dist <- proxy::dist(dtm_matrix, method = "cosine")

# Hierarchical clustering
hc <- hclust(cos_dist, method = "ward.D2")

# Convert to dendrogram
dend <- as.dendrogram(hc)

# Genre colours
genre_colours <- c(
  "Film Review" = "#E69F00",
  "News" = "#0072B2",
  "Sports" = "#009E73"
)

label_genres <- corpus_df$genre[match(labels(dend), corpus_df$doc_id)]

labels_colors(dend) <- genre_colours[label_genres]
labels_cex(dend) <- 0.9

# Save HD dendrogram
png(
  filename = "figures/dendrogram_hd.png",
  width = 2400,
  height = 1600,
  res = 250
)

par(mar = c(8, 5, 5, 2))

plot(
  dend,
  main = "Hierarchical Clustering of Documents Using Cosine Distance",
  ylab = "Height",
  xlab = "",
  sub = "Labels are coloured by actual genre"
)

rect.hclust(hc, k = 3, border = "red")

legend(
  "topright",
  legend = names(genre_colours),
  fill = genre_colours,
  border = NA,
  cex = 0.9,
  title = "Genre"
)

dev.off()

# Cluster assignment
cluster_assignments <- cutree(hc, k = 3)

cluster_results <- data.frame(
  doc_id = corpus_df$doc_id,
  title = corpus_df$title,
  genre = corpus_df$genre,
  cluster = cluster_assignments
)

write.csv(cluster_results, "data/cluster_results.csv", row.names = FALSE)

# Genre-cluster table
cluster_table <- table(cluster_results$genre, cluster_results$cluster)
print(cluster_table)

write.csv(
  as.data.frame.matrix(cluster_table),
  "data/cluster_genre_table.csv"
)

# Accuracy by majority genre
cluster_accuracy <- sum(apply(cluster_table, 2, max)) / sum(cluster_table)

cat("Clustering Accuracy:", round(cluster_accuracy * 100, 2), "%\n")

accuracy_df <- data.frame(
  metric = "Clustering accuracy by majority genre",
  value = round(cluster_accuracy * 100, 2)
)

write.csv(
  accuracy_df,
  "data/clustering_accuracy.csv",
  row.names = FALSE
)

install.packages("ggpubr")

# ============================================================
# QUESTION 5: SENTIMENT ANALYSIS
# ============================================================

library(syuzhet)
library(ggplot2)

# Get sentiment score for each document
sentiment_scores <- get_sentiment(
  corpus_df$text,
  method = "syuzhet"
)

# Create results table
sentiment_df <- data.frame(
  doc_id = corpus_df$doc_id,
  genre = corpus_df$genre,
  sentiment = sentiment_scores
)

print(sentiment_df)

# Save results
write.csv(
  sentiment_df,
  "data/sentiment_results.csv",
  row.names = FALSE
)

# Genre summary
genre_sentiment <- aggregate(
  sentiment ~ genre,
  data = sentiment_df,
  FUN = mean
)

print(genre_sentiment)

write.csv(
  genre_sentiment,
  "data/genre_sentiment.csv",
  row.names = FALSE
)

# Sentiment Boxplot
p1 <- ggplot(
    sentiment_df,
    aes(
        x = genre,
        y = sentiment,
        fill = genre
    )
) +
    geom_boxplot(alpha = 0.8) +
    geom_jitter(width = 0.1) +
    labs(
        title = "Sentiment Distribution by Genre",
        x = "Genre",
        y = "Sentiment Score"
    ) +
    theme_minimal() +
    theme(
        legend.position = "none",
        plot.title = element_text(face = "bold")
    )

ggsave(
    "figures/sentiment_boxplot.png",
    p1,
    width = 8,
    height = 5
)

# ============================================================
# ANOVA TEST
# ============================================================

anova_result <- aov(
  sentiment ~ genre,
  data = sentiment_df
)

summary(anova_result)

capture.output(
  summary(anova_result),
  file = "report/anova_results.txt"
)

library(dplyr)

sentiment_summary <- sentiment_df %>%
  group_by(genre) %>%
  summarise(
    mean_sentiment = mean(sentiment),
    sd_sentiment = sd(sentiment),
    min_sentiment = min(sentiment),
    max_sentiment = max(sentiment)
  )

print(sentiment_summary)

write.csv(
  sentiment_summary,
  "data/sentiment_summary.csv",
  row.names = FALSE
)
