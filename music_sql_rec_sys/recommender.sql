with all_users as ( 
  select user1_id as user_id, user2_id as friend_id from followers
  union
  select user2_id as user_id, user1_id as friend_id from followers
), follower_likes as (
  select au.user_id, l.song_id, count(l.user_id) as like_count
  from all_users as au 
  left join likes l on au.friend_id = l.user_id
  group by au.user_id, l.song_id
)
select fl.user_id, fl.song_id, fl.like_count, s.name, s.artist
from follower_likes as fl
left join likes as l on fl.user_id = l.user_id and fl.song_id = l.song_id
inner join songs as s on s.song_id = fl.song_id
where l.song_id is null
order by fl.user_id, fl.like_count desc