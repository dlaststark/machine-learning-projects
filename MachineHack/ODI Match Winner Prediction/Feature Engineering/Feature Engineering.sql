-- Table: machine_learning.odi_winner_predict

-- DROP TABLE machine_learning.odi_winner_predict;

CREATE TABLE machine_learning.odi_winner_predict
(
    "Team1" integer,
    "Team2" integer,
    "Stadium" integer,
    "HostCountry" integer,
    "Team1_Venue" character varying(20) COLLATE pg_catalog."default",
    "Team2_Venue" character varying(20) COLLATE pg_catalog."default",
    "Team1_Innings" character varying(20) COLLATE pg_catalog."default",
    "Team2_Innings" character varying(20) COLLATE pg_catalog."default",
    "MonthOfMatch" character varying(20) COLLATE pg_catalog."default",
    "MatchWinner" integer,
    "DatasetType" character varying(10) COLLATE pg_catalog."default"
)

TABLESPACE pg_default;

ALTER TABLE machine_learning.odi_winner_predict
    OWNER to postgres;
    
----------------------------------------------------------------------------------------------------------------------------------------------------------------

SELECT "Team1", "Team2", "Stadium", "HostCountry", "Team1_Venue", "Team2_Venue", "Team1_Innings", "Team2_Innings", "MonthOfMatch", "MatchWinner"
FROM machine_learning.odi_winner_predict;

----------------------------------------------------------------------------------------------------------------------------------------------------------------

-- #Matches
SELECT UNNEST(ARRAY["Team1", "Team2"]) AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
GROUP BY "Team" ORDER BY "Team";

----------------------------------------------------------------------------------------------------------------------------------------------------------------

-- #Wins
SELECT "MatchWinner" AS "Team", COUNT(1) AS "MatchesWon"
FROM machine_learning.odi_winner_predict
GROUP BY "Team" ORDER BY "Team";

----------------------------------------------------------------------------------------------------------------------------------------------------------------

-- #HomeVenueMatches
SELECT "Team", SUM("NumOfMatches") FROM (
SELECT "Team1" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team1_Venue" = 'Home'
GROUP BY "Team1"
UNION
SELECT "Team2" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team2_Venue" = 'Home'
GROUP BY "Team2") Base
GROUP BY "Team" ORDER BY "Team";

-- #AwayVenueMatches
SELECT "Team", SUM("NumOfMatches") FROM (
SELECT "Team1" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team1_Venue" = 'Away'
GROUP BY "Team1"
UNION
SELECT "Team2" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team2_Venue" = 'Away'
GROUP BY "Team2") Base
GROUP BY "Team" ORDER BY "Team";

-- #NeutralVenueMatches
SELECT "Team", SUM("NumOfMatches") FROM (
SELECT "Team1" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team1_Venue" = 'Neutral'
GROUP BY "Team1"
UNION
SELECT "Team2" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team2_Venue" = 'Neutral'
GROUP BY "Team2") Base
GROUP BY "Team" ORDER BY "Team";

----------------------------------------------------------------------------------------------------------------------------------------------------------------

-- #Home/Away/Neutral Venue Wins
SELECT "MatchWinner" AS "Team", SUM(
	CASE 
		WHEN "Team1" = "MatchWinner" AND "Team1_Venue" = 'Home' THEN 1
		WHEN "Team2" = "MatchWinner" AND "Team2_Venue" = 'Home' THEN 1
		ELSE 0
	END) AS "HomeWins",SUM(
	CASE 
		WHEN "Team1" = "MatchWinner" AND "Team1_Venue" = 'Away' THEN 1
		WHEN "Team2" = "MatchWinner" AND "Team2_Venue" = 'Away' THEN 1
		ELSE 0
	END) AS "AwayWins",SUM(
	CASE 
		WHEN "Team1" = "MatchWinner" AND "Team1_Venue" = 'Neutral' THEN 1
		WHEN "Team2" = "MatchWinner" AND "Team2_Venue" = 'Neutral' THEN 1
		ELSE 0
	END) AS "NeutralWins"
FROM machine_learning.odi_winner_predict
GROUP BY "Team" ORDER BY "Team";

----------------------------------------------------------------------------------------------------------------------------------------------------------------

-- #FirstInnings
SELECT "Team", SUM("NumOfMatches") FROM (
SELECT "Team1" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team1_Innings" = 'First'
GROUP BY "Team1"
UNION ALL
SELECT "Team2" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team2_Innings" = 'First'
GROUP BY "Team2") Base
GROUP BY "Team" ORDER BY "Team";

--#SecondInnings
SELECT "Team", SUM("NumOfMatches") FROM (
SELECT "Team1" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team1_Innings" = 'Second'
GROUP BY "Team1"
UNION ALL
SELECT "Team2" AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
WHERE "Team2_Innings" = 'Second'
GROUP BY "Team2") Base
GROUP BY "Team" ORDER BY "Team";

-- #First/Second Innings Wins
SELECT "MatchWinner" AS "Team", SUM(
	CASE 
		WHEN "Team1" = "MatchWinner" AND "Team1_Innings" = 'First' THEN 1
		WHEN "Team2" = "MatchWinner" AND "Team2_Innings" = 'First' THEN 1
		ELSE 0
	END) AS "FirstInningsWins",SUM(
	CASE 
		WHEN "Team1" = "MatchWinner" AND "Team1_Innings" = 'Second' THEN 1
		WHEN "Team2" = "MatchWinner" AND "Team2_Innings" = 'Second' THEN 1
		ELSE 0
	END) AS "SecondInningsWins"
FROM machine_learning.odi_winner_predict
GROUP BY "Team" ORDER BY "Team";

----------------------------------------------------------------------------------------------------------------------------------------------------------------

-- #Matches (month-wise)
SELECT "Team", 
SUM("JanMatches") AS "JanMatches",
SUM("FebMatches") AS "FebMatches",
SUM("MarMatches") AS "MarMatches",
SUM("AprMatches") AS "AprMatches",
SUM("MayMatches") AS "MayMatches",
SUM("JunMatches") AS "JunMatches",
SUM("JulMatches") AS "JulMatches",
SUM("AugMatches") AS "AugMatches",
SUM("SepMatches") AS "SepMatches",
SUM("OctMatches") AS "OctMatches",
SUM("NovMatches") AS "NovMatches",
SUM("DecMatches") AS "DecMatches"
FROM (
SELECT UNNEST(ARRAY["Team1", "Team2"]) AS "Team", 
CASE WHEN "MonthOfMatch" = 'Jan' THEN 1 END AS "JanMatches",
CASE WHEN "MonthOfMatch" = 'Feb' THEN 1 END AS "FebMatches",
CASE WHEN "MonthOfMatch" = 'Mar' THEN 1 END AS "MarMatches",
CASE WHEN "MonthOfMatch" = 'Apr' THEN 1 END AS "AprMatches",
CASE WHEN "MonthOfMatch" = 'May' THEN 1 END AS "MayMatches",
CASE WHEN "MonthOfMatch" = 'Jun' THEN 1 END AS "JunMatches",
CASE WHEN "MonthOfMatch" = 'Jul' THEN 1 END AS "JulMatches",
CASE WHEN "MonthOfMatch" = 'Aug' THEN 1 END AS "AugMatches",
CASE WHEN "MonthOfMatch" = 'Sep' THEN 1 END AS "SepMatches",
CASE WHEN "MonthOfMatch" = 'Oct' THEN 1 END AS "OctMatches",
CASE WHEN "MonthOfMatch" = 'Nov' THEN 1 END AS "NovMatches",
CASE WHEN "MonthOfMatch" = 'Dec' THEN 1 END AS "DecMatches"
FROM machine_learning.odi_winner_predict) Base
GROUP BY "Team" ORDER BY "Team";

----------------------------------------------------------------------------------------------------------------------------------------------------------------

-- %Wins (month-wise)
SELECT "Team", 
SUM("JanMatches") AS "JanMatches",
SUM("FebMatches") AS "FebMatches",
SUM("MarMatches") AS "MarMatches",
SUM("AprMatches") AS "AprMatches",
SUM("MayMatches") AS "MayMatches",
SUM("JunMatches") AS "JunMatches",
SUM("JulMatches") AS "JulMatches",
SUM("AugMatches") AS "AugMatches",
SUM("SepMatches") AS "SepMatches",
SUM("OctMatches") AS "OctMatches",
SUM("NovMatches") AS "NovMatches",
SUM("DecMatches") AS "DecMatches"
FROM (
SELECT "MatchWinner" AS "Team", 
CASE WHEN "MonthOfMatch" = 'Jan' THEN 1 END AS "JanMatches",
CASE WHEN "MonthOfMatch" = 'Feb' THEN 1 END AS "FebMatches",
CASE WHEN "MonthOfMatch" = 'Mar' THEN 1 END AS "MarMatches",
CASE WHEN "MonthOfMatch" = 'Apr' THEN 1 END AS "AprMatches",
CASE WHEN "MonthOfMatch" = 'May' THEN 1 END AS "MayMatches",
CASE WHEN "MonthOfMatch" = 'Jun' THEN 1 END AS "JunMatches",
CASE WHEN "MonthOfMatch" = 'Jul' THEN 1 END AS "JulMatches",
CASE WHEN "MonthOfMatch" = 'Aug' THEN 1 END AS "AugMatches",
CASE WHEN "MonthOfMatch" = 'Sep' THEN 1 END AS "SepMatches",
CASE WHEN "MonthOfMatch" = 'Oct' THEN 1 END AS "OctMatches",
CASE WHEN "MonthOfMatch" = 'Nov' THEN 1 END AS "NovMatches",
CASE WHEN "MonthOfMatch" = 'Dec' THEN 1 END AS "DecMatches"
FROM machine_learning.odi_winner_predict) Base
GROUP BY "Team" ORDER BY "Team";

----------------------------------------------------------------------------------------------------------------------------------------------------------------

-- #Matches/Wins (stadium-wise)
SELECT "Team", foo."Stadium", "NumOfMatches", COALESCE("NumOfWins",0) AS "NumOfWins" 
FROM
(SELECT UNNEST(ARRAY["Team1", "Team2"]) AS "Team", "Stadium", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
GROUP BY "Team", "Stadium" ORDER BY "Team", "Stadium") foo
LEFT JOIN
(SELECT "MatchWinner", "Stadium", COUNT(1) AS "NumOfWins"
FROM machine_learning.odi_winner_predict
GROUP BY "MatchWinner", "Stadium" ORDER BY "MatchWinner", "Stadium") bar
ON foo."Team" = bar."MatchWinner" AND foo."Stadium" = bar."Stadium"
ORDER BY "Team", foo."Stadium";

----------------------------------------------------------------------------------------------------------------------------------------------------------------

-- #Match Wins (Team-vs-Team)
WITH Base AS (
SELECT foo."Team"[1] AS "Team1", foo."Team"[2] AS "Team2", "NumOfMatches", "MatchWinner", "NumOfWins"
FROM
(SELECT "Team", SUM("NumOfMatches") AS "NumOfMatches" FROM (
SELECT SORT(ARRAY["Team1", "Team2"]) AS "Team", COUNT(1) AS "NumOfMatches"
FROM machine_learning.odi_winner_predict
GROUP BY "Team1", "Team2") Base GROUP BY "Team") foo
JOIN
(SELECT "Team", "MatchWinner", SUM("NumOfWins") AS "NumOfWins"
FROM (
SELECT SORT(ARRAY["Team1", "Team2"]) AS "Team", "MatchWinner", COUNT(1) AS "NumOfWins"
FROM machine_learning.odi_winner_predict
GROUP BY "Team1", "Team2", "MatchWinner" ORDER BY "Team1", "Team2") Base
GROUP BY "Team", "MatchWinner") bar
ON foo."Team" = bar."Team"
ORDER BY foo."Team"[1], foo."Team"[2])

SELECT "Team1", "Team2", "NumOfMatches",
SUM(CASE WHEN "Team1" = "MatchWinner" THEN "NumOfWins" ELSE 0 END) AS "Team1Win",
SUM(CASE WHEN "Team2" = "MatchWinner" THEN "NumOfWins" ELSE 0 END) AS "Team2Win"
FROM Base GROUP BY "Team1", "Team2", "NumOfMatches" ORDER BY "Team1", "Team2", "NumOfMatches";

