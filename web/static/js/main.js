/*
    CodAI - Programming language Detection AI
    Copyright(C) 2018 Rupesh Sreeraman
*/
"use strict";

function LoadModel() {
 google.charts.load("current", {packages:["corechart"]});
$("#detectBtn").prop('disabled', false);
 $("#messageText").prop('disabled', false);
}
function checkInput()
{
	var x = document.getElementById("messageText").value;
	if (x=="")
	{   
        $("#detectBtn").prop('disabled', true);
        $("#messageType").html("<div class=\"alert alert-warning\">Waiting for input...</div>");
       
		return;
	}
	else{
		$("#detectBtn").prop('disabled', false);
		
	}
}
function predictCodeLanguage() {
	
	var x = document.getElementById("messageText").value;
	if (x=="")
	{ 
      return;
	}
	
	NProgress.start();
	
	$.ajax({
    type : "POST",
    url : "predict",
    data: JSON.stringify(x),
    contentType: 'application/json;charset=UTF-8',
    success: function(result) {
        //console.log(result);
		var probs=[];
   var languages=[];
		var classProb=eval(result);
	
		classProb.predictions.sort(function(a, b){
    return  b.probability-a.probability;
      });
   var sum=0;

   var langs="";
   
 
   
     var msg="<div><h4>Code seems like <span class=\"badge badge-pill badge-primary  text-capitalize\"> "+ classProb.predictions[0].label+" </span></h4></div>";
     var subMsg="<div class=\"text-muted\">Other possible languages <div class=\"badge badge-pill badge-dark text-capitalize\">"+classProb.predictions[1].label+"</div> <div <div class=\"badge badge-pill badge-secondary text-capitalize\">"+classProb.predictions[2].label+"</div></div>";
     $("#messageType").html(msg);
	 $("#otherLanguages").html(subMsg);
		//console.log(langs);
		
		
  	for (var i=0;i<classProb.predictions.length;i++)
   {
	   probs.push(classProb.predictions[i].probability);
	   languages.push(classProb.predictions[i].label);
   }

		
		var data = new google.visualization.DataTable();
            // assumes "word" is a string and "count" is a number
            data.addColumn('string', 'Language');
            data.addColumn('number', 'Score');

            for (var i = 0; i < languages.length; i++) {
                data.addRow([String(languages[i]), Number(probs[i])]);
            }

            var options = {
                title: 'Language score',
                is3D: false
            };
            var chart = new google.visualization.PieChart(document.getElementById('piechart_3d'));
            chart.draw(data, options);
			NProgress.done();
    }
});
	
	
	
	

}