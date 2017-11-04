// 梯度下降
var m = -1000;
var b = 1000;
data = [
    {x: 1, y: 1},
    {x: 2, y: 2},
    {x: 3, y: 3},
    {x: 4, y: 4}
];

function grad() {
    var learning_rate = 0.05;
    for (var i = 0; i < data.length; i++) {
        var x = data[i].x;
        var y = data[i].y;
        var guess = m * x + b;
        var error = y - guess;
        m = m + error * x * learning_rate;
        b = b + error * learning_rate;
    }
}

for (var i = 0; i < 1000; i++)
    grad()

console.log(m, b);