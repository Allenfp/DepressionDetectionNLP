
// Read the text and output the text.

function myFunction() {
    var input = document.getElementById('inputText');
    
    let output = document.getElementById('outputText');
    
    output.innerHTML = "Result: " + input.value;
}