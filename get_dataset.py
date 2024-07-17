import json
import polars as pl

pl.Config.set_fmt_str_lengths(50)

def prepare_dataset():
    movies = pl.read_csv("warsztat_data.csv", infer_schema_length=10000)
    # tmdb = movies.select("title", "overview", "release_date", "genres", "id").join(
    #     credits.select("cast", "movie_id"), left_on="id", right_on="movie_id"
    # )
    # return movies
    tmdb_texts = movies.select(
        "id",
        pl.struct(["nazwa_usterki" ,"opis_usterki", "model_samochodu"])
        .apply(
            lambda row: f"""passage: Nazwa usterki {row['nazwa_usterki']}
    Opis usterki: {row['opis_usterki']}
    Model samochodu: {row['model_samochodu']}""".strip()
        )
        .alias("text"),
        )

    return {"text": tmdb_texts["text"].to_list(), "metadata": [{"id": text_id} for text_id in tmdb_texts["id"]]}

# print(prepare_dataset())

# def prepare_dataset():
#     def movie_cast_as_text(cast_json, top_n=7):
#         cast = json.loads(cast_json)
#         formatted = ["- {role} played by {actor}. ".format(role=c["character"], actor=c["name"]) for c in cast[:top_n]]
#         return "Cast:\n" + "\n".join(formatted)

#     def movie_genres_as_text(genres_json, top_n=3):
#         genres = json.loads(genres_json)
#         formatted = "Movie genres:\n" + "\n".join(f"- {g['name'].lower()}" for g in genres[:top_n])
#         return formatted
    
#     credits = pl.read_csv("tmdb_5000_credits.csv")
#     movies = pl.read_csv("tmdb_5000_movies.csv", infer_schema_length=10000)
#     tmdb = movies.select("title", "overview", "release_date", "genres", "id").join(
#         credits.select("cast", "movie_id"), left_on="id", right_on="movie_id"
#     )

#     tmdb_texts = tmdb.select(
#         "id",
#         pl.struct(["title", "overview", "release_date", "genres", "cast"])
#         .apply(
#             lambda row: f"""passage: Title: {row['title']}
#     Summary: {row['overview']}
#     Released on: {row['release_date']}
#     {movie_cast_as_text(row['cast'])}
#     {movie_genres_as_text(row['genres'])}
#     """.strip()
#         )
#         .alias("text"),
#     )
#     # return tmdb_texts
#     return {"text": tmdb_texts["text"].to_list(), "metadata": [{"id": text_id} for text_id in tmdb_texts["id"]]}