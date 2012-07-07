# favoriteblogs.R
# finds peoples' favorite blogs and predicts that they will like other posts on them

t <- proc.time()
library(rjson)

# Read in the test data
testPostsThinLines <- readLines("testPostsThin.json")

blog_id <- ""
blog_posts <- list()

line_count <- 0
for(line in testPostsThinLines){
	dejsoned <- fromJSON(line)
	post_id <- dejsoned$post_id

	if(blog_id != dejsoned$blog){
		blog_id <- dejsoned$blog
		blog_posts[[blog_id]] <- vector()
	}
	blog_posts[[blog_id]] <- c(blog_posts[[blog_id]], post_id)

	line_count <- line_count + 1
	if(line_count %% 20000 == 0){
		cat(line_count, "/ 236282\n")
	}
}
proc.time() - t # about 6 minutes since start

# Generate top posts, for users who don't like very many blogs
blogStatsLines <- readLines("kaggle-stats-blogs-20111123-20120423.json")

blog_ids <- vector()
blog_scores <- numeric()
blog_count <- 0

line_count <- 0
for(line in blogStatsLines){
	dejsoned <- fromJSON(line)
	blog_id <- dejsoned$blog_id
	num_posts <- as.numeric(dejsoned$num_posts)
	num_likes <- as.numeric(dejsoned$num_likes)
	
	blog_count <- blog_count + 1
	blog_ids[blog_count] <- blog_id
	blog_scores[blog_count] <- num_likes / (num_posts + 10)

	line_count <- line_count + 1
	if(line_count %% 20000 == 0){
		cat(line_count, "/ 86576\n")
	}
}

top_blogs <- blog_ids[sort(blog_scores, index.return=TRUE, decreasing=TRUE)$ix]
top_posts <- vector()
for(blog_id in top_blogs[1:50]){
	top_posts <- c(top_posts, blog_posts[[as.character(blog_id)]])
}
proc.time() - t # about 9 minutes since start

# Make predictions based upon users' favorite blogs, or in lieu of that, the most popular blog posts
outputFile <- "favoriteblogs.csv"
cat("\"posts\"\n", file=outputFile, append=FALSE)

trainUserLines <- readLines("trainUsers.json")
line_count <- 0
for(line in trainUserLines){
	line_count <- line_count + 1
	if(line_count %% 10000 == 0){
		cat(line_count, "/ 86758\n")
	}

	dejsoned <- fromJSON(line)
	if(dejsoned$inTestSet == FALSE){
		next
	}

	likes <- dejsoned$likes
	liked_blogs <- vector()
	liked_posts <- vector()

	for(like in likes){
		liked_blogs <- c(liked_blogs, like$blog)
		liked_posts <- c(liked_posts, like$post_id)
	}
	favorite_blogs <- names(sort(table(liked_blogs), decreasing=TRUE))
	favorite_blogs <- favorite_blogs[1:(min(length(favorite_blogs), 3))]

	post_recs <- vector()
	for(blog_id in favorite_blogs){
		post_recs <- c(post_recs, blog_posts[[blog_id]])
	}
	
	if(length(post_recs) < 100){
		post_recs <- c(post_recs, top_posts[1:(100 - length(post_recs))])
	}
	else {
		post_recs <- post_recs[1:100]
	}
	
	lineOut <- ""
	for(post_id in post_recs){
		lineOut <- paste(lineOut, post_id, " ", sep="")
	}
	lineOut <- paste(lineOut, "\n", sep="")
	cat(lineOut, file=outputFile, append=TRUE)

}
proc.time() - t # about 11 minutes since start

