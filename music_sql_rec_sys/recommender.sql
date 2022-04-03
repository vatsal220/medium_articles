with all_users (user_id, friend_id) as (
   select
      user1_id,
      user2_id
   from followers
   union
   select
      user1_id,
      user2_id
   from followers
), follower_likes (user_id, song_id, name, artist, like_count) as (
   select
      au.user_id, 
      likes.song_id,
      songs.name,
      songs.artist,
      count(*) as like_count
   from all_users as au
   inner join likes on likes.user_id = au.user_id
   inner join songs on songs.song_id = likes.song_id
   group by au.user_id, likes.song_id, songs.name, songs.artist
)

select
   fl.user_id,
   fl.song_id,
   fl.like_count,
   fl.name,
   fl.artist
from follower_likes as fl
left join likes on likes.user_id = fl.user_id and fl.song_id = likes.song_id
where likes.song_id is null




