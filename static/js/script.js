document.addEventListener("DOMContentLoaded",function(){

const loader=document.getElementById("loader");

window.addEventListener("load",()=>{

setTimeout(()=>{

loader.classList.add("hide");

},700);

});

const uploadForm=document.querySelector("form[enctype='multipart/form-data']");

if(uploadForm){

uploadForm.addEventListener("submit",()=>{

loader.classList.remove("hide");

});

}

const links=document.querySelectorAll('a[href^="#"]');

links.forEach(link=>{

link.addEventListener("click",function(e){

e.preventDefault();

const target=document.querySelector(this.getAttribute("href"));

if(target){

target.scrollIntoView({

behavior:"smooth"

});

}

});

});

const sections=document.querySelectorAll("section");

const navLinks=document.querySelectorAll(".nav-link");

window.addEventListener("scroll",()=>{

let current="";

sections.forEach(section=>{

const top=section.offsetTop-120;

if(window.scrollY>=top){

current=section.id;

}

});

navLinks.forEach(link=>{

link.classList.remove("active");

if(link.getAttribute("href")==="#"+current){

link.classList.add("active");

}

});

});

const observer=new IntersectionObserver(entries=>{

entries.forEach(entry=>{

if(entry.isIntersecting){

entry.target.style.opacity="1";

entry.target.style.transform="translateY(0)";

}

});

},{threshold:.15});

document.querySelectorAll(".glass-card,.dashboard-card").forEach(card=>{

card.style.opacity="0";

card.style.transform="translateY(50px)";

card.style.transition=".8s";

observer.observe(card);

});

document.querySelectorAll(".btn").forEach(button=>{

button.addEventListener("click",function(e){

const ripple=document.createElement("span");

const rect=this.getBoundingClientRect();

const size=Math.max(rect.width,rect.height);

ripple.style.width=size+"px";

ripple.style.height=size+"px";

ripple.style.left=e.clientX-rect.left-size/2+"px";

ripple.style.top=e.clientY-rect.top-size/2+"px";

ripple.className="ripple";

this.appendChild(ripple);

setTimeout(()=>{

ripple.remove();

},600);

});

});

const back=document.getElementById("backToTop");

window.addEventListener("scroll",()=>{

if(window.scrollY>400){

back.classList.add("show");

}else{

back.classList.remove("show");

}

const winScroll=document.body.scrollTop||document.documentElement.scrollTop;

const height=document.documentElement.scrollHeight-document.documentElement.clientHeight;

const scrolled=(winScroll/height)*100;

document.getElementById("progressBar").style.width=scrolled+"%";

});

back.addEventListener("click",()=>{

window.scrollTo({

top:0,

behavior:"smooth"

});

});

});