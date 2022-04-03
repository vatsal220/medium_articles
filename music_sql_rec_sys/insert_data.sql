create table followers (
  user1_id int,
  user2_id int
);

create table likes (
  user_id int,
  song_id int
);
create table songs (
  song_id int,
  name varchar,
  artist varchar
);

insert into followers values ('1', '2');
insert into followers values ('1', '3');
insert into followers values ('1', '4');
insert into followers values ('2', '1');
insert into followers values ('2', '3');
insert into followers values ('2', '4');
insert into followers values ('3', '2');
insert into followers values ('4', '2');


insert into likes values ('1', '1');
insert into likes values ('1', '4');
insert into likes values ('1', '5');
insert into likes values ('2', '2');
insert into likes values ('2', '3');
insert into likes values ('2', '7');
insert into likes values ('2', '9');
insert into likes values ('2', '11');
insert into likes values ('2', '12');
insert into likes values ('3', '13');
insert into likes values ('3', '6');
insert into likes values ('3', '8');
insert into likes values ('3', '3');
insert into likes values ('3', '2');
insert into likes values ('4', '1');
insert into likes values ('4', '8');
insert into likes values ('4', '3');

insert into songs values ('1', 'pipe down', 'drake');
insert into songs values ('2', 'sanctuary', 'joji');
insert into songs values ('3', 'gold digger', 'kanye');
insert into songs values ('4', 'butter', 'bts');
insert into songs values ('5', 'ultimate', 'denzel curry');
insert into songs values ('6', 'daily routine', 'joey badass');
insert into songs values ('7', 'self love', 'mavi');
insert into songs values ('8', 'chanel', 'frank ocean');
insert into songs values ('9', 'spaceship', 'kanye');
insert into songs values ('10', 'tuscan leather', 'drake');
insert into songs values ('11', 'homecoming', 'kanye');
insert into songs values ('12', 'hello', 'adele');
insert into songs values ('13', 'easy on me', 'adele');