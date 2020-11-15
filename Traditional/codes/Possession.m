function Possession

eventTable = readtable('fullevents.csv');
eventTable(1,1);
maxSize = size(eventTable);
maxRow = maxSize(1);
match = 1;
ansTable = table;
ansTable.MatchID = zeros(38,1);
ansTable.HuskiesTime1H = zeros(38,1);
ansTable.OpponentTime1H = zeros(38,1);
ansTable.HuskiesTime2H = zeros(38,1);
ansTable.OpponentTime2H = zeros(38,1);
ansTable.totTime = zeros(38,1);
ansTable.HuskiesPossess = zeros(38,1);
ansTable.OppoPossess = zeros(38,1);
HuskiesTime = 0;
OpponentTime = 0;
tempEventTime = eventTable.EventTime(1);
tempTeamID = {'Haa'};
tempPeriod = 0;
addTime = 0;
row = 0;
while (row < maxRow)
    row = row + 1;
    tt = eventTable.MatchPeriod(row);
    if (tt{1}(1) ~= tempPeriod)
        if (row > 1)
            addTime = eventTable.EventTime(row-1) - tempEventTime;
        end
        if (tempTeamID{1}(1) == 'H')
            HuskiesTime = HuskiesTime + addTime;
        else
            OpponentTime = OpponentTime + addTime;
        end        
        if (match ~= eventTable.MatchID(row))
            ansTable.MatchID(match) = match;
            ansTable.HuskiesTime2H(match) = HuskiesTime;
            ansTable.OpponentTime2H(match) = OpponentTime;
            HuskiesTime = 0;
            OpponentTime = 0;
            match = eventTable.MatchID(row);
        else
            ansTable.HuskiesTime1H(match) = HuskiesTime;
            ansTable.OpponentTime1H(match) = OpponentTime;
            HuskiesTime = 0;
            OpponentTime = 0;
        end
        tempEventTime = eventTable.EventTime(row);
        tempTeamID = eventTable.TeamID(row);
        tempPeriod = tt{1}(1);
    else
        t = eventTable.TeamID(row);
        if (row < maxRow)
            tempEventType = eventTable.EventType(row);
            nextEventType = eventTable.EventType(row + 1);
            tempTeamID2 = eventTable.TeamID(row);
            nextTeamID2 = eventTable.TeamID(row + 1);
            if (tempEventType{1}(1) == 'D' && nextEventType{1}(1) == 'D' ...
                    && tempTeamID2{1}(1) ~= nextTeamID2{1}(1)...
                    && (eventTable.EventOrigin_x(row) + eventTable.EventOrigin_x(row + 1) == 100) ...
                    && (eventTable.EventOrigin_y(row) + eventTable.EventOrigin_y(row + 1) == 100))
                row = row + 1;
            else
                if (t{1}(2) ~= tempTeamID{1}(2))               
                    addTime = eventTable.EventTime(row) - tempEventTime;
                    if (tempTeamID{1}(1) == 'H')
                        HuskiesTime = HuskiesTime + addTime;
                    else
                        OpponentTime = OpponentTime + addTime;
                    end
                    tempEventTime = eventTable.EventTime(row);
                    tempTeamID = eventTable.TeamID(row);
                end
            end
        end
    end
end
addTime = eventTable.EventTime(maxRow) - tempEventTime;
if (tempTeamID{1}(1) == 'H')
    HuskiesTime = HuskiesTime + addTime;
else
    OpponentTime = OpponentTime + addTime;
end
ansTable.MatchID(match) = match;
ansTable.HuskiesTime2H(match) = HuskiesTime;
ansTable.OpponentTime2H(match) = OpponentTime;
writetable(ansTable,'C:\Users\zhbq\Documents\MATLAB\direct_stat2.csv','Delimiter',',','QuoteStrings',true);
end