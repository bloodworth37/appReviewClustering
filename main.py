# This is the central script which will execute each
# subscript in sequence

# attributes:
# reviewId, userName, userImage, content, score, thumbsUpCount
# reviewCreatedVersion, at, replyContent, repliedAt, sortOrder, appId

from genEmbeddings import generate_embeddings

generate_embeddings("20reviews.csv")