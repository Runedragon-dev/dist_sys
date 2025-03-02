FROM golang:1.23-alpine AS builder

WORKDIR /app

WORKDIR /app/datamodel
COPY datamodel/go.mod datamodel/go.sum ./
RUN go mod download
COPY datamodel/ ./
RUN go build -o /app/bin/datamodel ./ 

WORKDIR /app/rest
COPY rest/go.mod rest/go.sum ./
RUN go mod download
COPY rest/ ./
RUN go build -o /app/bin/rest ./ 

WORKDIR /app/mlwebapi
COPY mlwebapi/go.mod mlwebapi/go.sum ./
RUN go mod download
COPY mlwebapi/ ./
RUN go build -o /app/bin/mlwebapi ./

FROM alpine:latest

WORKDIR /root/
COPY --from=builder /app/bin/datamodel .
COPY --from=builder /app/bin/rest .
COPY --from=builder /app/bin/mlwebapi .
#COPY . .

COPY mlwebapi/ModelsTokens.txt .
COPY mlwebapi/ModelsURLs .

CMD ["./mlwebapi"]