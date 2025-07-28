## Learnings.
- Give ans to user question based on data stored in a SQL DB and we are going to use Agents for this.
- While querying users can make spelling mistake so we have to store some common noun in Vector-db so that LLM can use that to determine if the spelling is right or wrong which will help LLM to write proper query to get correct data from DB.


- Run this command to get the DB.
```shell
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db
```