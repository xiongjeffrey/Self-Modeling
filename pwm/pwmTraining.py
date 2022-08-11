from perfectWM import *

# import dataset here
# take a predator-prey environment and simulate it, collecting 2 datapoints at each time interval (one for prey, one for predator)
deathData = []
goalData  = []

with open('pwm\\pwmData.csv', 'r') as csvfile: # courtesy of https://stackoverflow.com/questions/13428318/reading-rows-from-a-csv-file-in-python

    r = csv.reader(csvfile, delimiter = ' ', quotechar = '|', quoting = csv.QUOTE_MINIMAL)
    for _, line in enumerate(r):
        deathData.append([int(line[0][1]), int(line[0][4])])
        goalData.append([int(line[1][1]), int(line[1][4])])

# create NNs
deathNet = DeathNN()
goalNet  = GoalNN()

deathEnc = []
goalEnc  = []
for i in range(len(deathData)):
    deathEnc.append(encode(deathData[i], width, height))
    goalEnc.append(encode(goalData[i], width, height))

# learn the NNs
def train(net, data): # target for data[t] is data[t + 1]
    
    # boilerplate optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.01)
    criterion = nn.MSELoss() #!TODO: figure out proper loss function to use

    # # tracking running + loss
    # running_loss = 0
    # start_time = time.time()

    print('Training network...')

    # looping through for 10 epochs
    for i in range(10):

        for j in range(1000):

            input = data[j]
            target = decode(data[j + 1]).view(1, 2).float()

            optimizer.zero_grad()
            output = net(input)

            output = output.view(-1, 2) 
            loss   = criterion(output, target)

            loss.backward()
            optimizer.step()

            # running_loss += loss.item()

            # if j % 100 == 99:
            #     running_loss /= 100*(len(data) - 1)
            #     print('Step {}, Loss {:0.10f}, Time{:0.1f}s'.format(i + 1, running_loss, time.time() - start_time))
            #     running_loss = 0

    return net

deathNet = train(deathNet, deathEnc)
goalNet = train(goalNet, goalEnc)

torch.save(deathNet.state_dict(), 'predNN.pt')
torch.save(goalNet.state_dict(), 'goalNN.pt')